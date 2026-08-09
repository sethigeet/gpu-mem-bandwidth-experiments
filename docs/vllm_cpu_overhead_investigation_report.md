# vLLM CPU and Python Overhead Investigation

## Objective

Identify where wall-clock time goes in vLLM serving for a 10K-token
shared-prefix workload, distinguish CPU/frontend overhead from GPU execution,
and identify concrete optimization targets.

## Configuration

- vLLM 0.22.1 on an NVIDIA RTX 6000 Ada Generation
- `meta-llama/Llama-3.1-8B`, float16, FlashAttention
- 10,000-token shared prefix + 20 unique input tokens
- 50 generated tokens per request
- 300 requests, maximum concurrency and `max_num_seqs` 100
- FCFS scheduling with prefix caching
- one async and one sync trial
- Nsight Systems GPU metrics at 500 Hz
- 60-second py-spy subprocess capture with native frames

The main instrumented run simultaneously enabled cumulative function timing,
py-spy, and NSYS collection. Its throughput therefore measures the profiled
system, not uninstrumented production performance.

## Experiment status

| Experiment | Status | Result |
| --- | --- | --- |
| A: timing + py-spy + NSYS | Partial success | All requests and traces completed; py-spy captured 5,012 async and 5,513 sync samples, but fell 24-26 seconds behind |
| B: online versus offline | Success after retry | Initial FlashInfer sampler JIT failed; rerun with `VLLM_USE_FLASHINFER_SAMPLER=0` completed |
| C: torch-profiler window | Success after retry | The old profiler environment variable produced no trace; `--profiler-config` generated a 30.8 MB trace and summary |
| D: fresh GPU-busy analysis | Success after correction | Two fresh NSYS traces analyzed; GR-active extraction was corrected to exclude cycle-count metrics |
| E: unaccounted-step drilldown + GPU timing | Success after profiler workaround | Exclusive step regions resolved the remainder; NSYS captured 17K+ kernels per mode, and CUDA events supplied per-call GPU duration after NSYS's child-process projection failed |

## Instrumented serving results

| Mode | Output throughput (tok/s) | Duration (s) | Mean TTFT (ms) | Mean TPOT (ms) | DRAM mean % | DRAM p95 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sync | 309.96 | 48.39 | 11,098.63 | 67.97 | 5.78 | 52 |
| Async | 310.87 | 48.25 | 12,498.82 | 35.35 | 5.67 | 46 |

Async throughput was only 0.29% higher. It halved mean TPOT but increased mean
TTFT by 12.6%, reflecting pipeline latency and the heavily instrumented,
always-backlogged workload.

## Per-iteration EngineCore budget

`GPUModelRunner._prepare_inputs` is nested inside `execute_model`; the table
subtracts it from execute time to avoid double counting. Async input-preparation
totals are normalized over all 500 EngineCore steps even though only 400 steps
called the preparation function.

| Core phase | Sync ms/step | Sync % | Async ms/step | Async % |
| --- | ---: | ---: | ---: | ---: |
| Scheduler | 9.02 | 9.6 | 8.26 | 10.6 |
| Input preparation | 14.67 | 15.6 | 11.59 | 14.8 |
| Execute model, excluding preparation | 25.39 | 27.0 | 30.94 | 39.6 |
| Update from output | 1.47 | 1.6 | 1.77 | 2.3 |
| Unaccounted inside step | 43.44 | 46.2 | 25.54 | 32.7 |
| **EngineCore step** | **93.99** | **100.0** | **78.11** | **100.0** |

Async reduced measured EngineCore step time by 16.9%. Most of the reduction was
in the unaccounted remainder and input preparation, while model execution
excluding preparation increased.

### Follow-up: resolving the unaccounted remainder

The follow-up separated the profilers and added exclusive regions directly
inside `EngineCore.step` and `EngineCore.step_with_batch_queue`. This run used
the same model and 300-request workload but omitted py-spy and NSYS from the CPU
phase measurement, raising throughput to 707.69 tok/s sync and 721.51 tok/s
async. Its absolute times therefore should not be mixed with Experiment A, but
the exclusive breakdown identifies what the previous subtraction called
"unaccounted."

| Exclusive EngineCore phase | Sync ms/step | Sync % | Async ms/step | Async % |
| --- | ---: | ---: | ---: | ---: |
| Scheduler | 1.25 | 1.43 | 1.56 | 2.13 |
| Execute-model submission | 7.61 | 8.71 | 6.54 | 8.96 |
| Sync fallback `sample_tokens` | 78.82 | 90.23 | — | — |
| Async initial sample submission | — | — | 6.73 | 9.23 |
| Async model/sample future wait | — | — | 58.99 | 80.82 |
| Update from output | 0.31 | 0.36 | 0.39 | 0.53 |
| Request checks, grammar, queues, and abort handling | 0.01 | 0.01 | 0.01 | 0.02 |
| Timer reconciliation error | -0.64 | -0.74 | -1.23 | -1.69 |
| **EngineCore step** | **87.36** | **100.0** | **72.99** | **100.0** |

The sync remainder is overwhelmingly the fallback call to
`model_executor.sample_tokens()` after `execute_model()` returns `None`. In the
async batch-queue path, the corresponding cost appears mainly in
`future.result()`, with smaller launch-side costs for model and sample
submission. The small negative reconciliation row is timing/context-manager
instrumentation overhead, not another execution phase.

![Exclusive EngineCore step breakdown](../results/vllm_cpu_overhead_drilldown_20260809/step_breakdown.png)

Frontend output processing is in a separate process and can overlap EngineCore,
so it is not additive with the table:

| Frontend function | Sync | Async |
| --- | ---: | ---: |
| `OutputProcessor.process_outputs` per call | 42.74 ms (500 calls) | 46.61 ms (400 calls) |
| Output-processing total | 21.37 s | 18.65 s |
| `BaseIncrementalDetokenizer.update` per token update | 1.25 ms | 1.27 ms |
| Detokenization total, 15,400 calls | 19.24 s | 19.50 s |

Detokenization consumed almost as much cumulative time as the complete output
processor. The totals are cumulative function times and may overlap across
threads/processes; they should not be summed with EngineCore wall time.

## py-spy findings

The APIServer processes were PID 1148623 (sync) and 1147583 (async). Their
largest native self-time frames were:

| APIServer hotspot | Sync self time | Sync share | Async self time | Async share |
| --- | ---: | ---: | ---: | ---: |
| Tokenizer flatten/iteration | 2.79 s | 8.42% | 2.60 s | 7.85% |
| `malloc` | 1.55 s | 4.68% | 1.75 s | 5.28% |
| Token ID to token conversion | 1.21 s | 3.65% | 1.03 s | 3.11% |
| Tokenizer hash-map hashing | 0.96 s | 2.90% | 0.97 s | 2.93% |
| `free` | 0.91 s | 2.75% | 0.78 s | 2.36% |

This corroborates the timing hooks: frontend token conversion, tokenizer data
structures, and allocation are material costs.

The EngineCore processes were PID 1148665 (sync) and 1147626 (async). Native
CUDA-driver frames dominated their top entries. The identifiable CPU hotspots
included:

- SHA-256 block hashing: 0.30 s sync and 0.37 s async;
- `find_longest_cache_hit`: 0.23 s async;
- NumPy packing and integer writes; and
- request block hashing and MessagePack decoding on the input-socket thread.

The input-socket thread's largest self-time frame was SHA-256: 15.7% of that
thread's sync samples and 20.4% of its async samples.

These flamegraphs are qualitative. py-spy reported 19-23 sampling errors and
fell 24-26 seconds behind, so percentages and temporal alignment are not precise.

## GPU activity and idle time

The corrected fresh NSYS analysis used `GR Active [Throughput %]` as the busy
proxy and excluded `GR Active [Cycles Active]`.

| Mode | GR active mean % | SM active mean % | Median % | Samples below 10% | Samples above 80% | DRAM mean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sync | 24.44 | 22.11 | 0 | 74.21% | 23.37% | 5.78 |
| Async | 27.54 | 24.58 | 0 | 71.70% | 26.95% | 5.67 |

The GPU was below 10% activity in roughly 72-74% of the active DRAM window.
Async increased mean GR activity by 3.10 percentage points, but did not increase
instrumented throughput materially. The workload is bursty and GPU-idle
dominated at this measurement boundary, not continuously memory-bandwidth-bound.

### Follow-up: CPU submission time versus GPU execution time

Separate NSYS runs completed at 712.96 tok/s sync and 740.20 tok/s async and
captured 17,508 and 17,407 kernels, respectively. The intended NSYS NVTX GPU
projection could not be used: NSYS 2026.1 recorded the EngineCore child
process's CUDA timestamps in a different time origin from its NVTX timestamps,
so NVIDIA's `nvtx_gpu_proj_sum` report returned no range rows despite both the
ranges and kernels being present in SQLite.

The confirmation pass therefore retained the same NVTX method boundaries and
placed CUDA events on the model runner's stream around
`GPUModelRunner.execute_model` and `_prepare_inputs`. It completed at 714.93
tok/s sync and 741.27 tok/s async.

| Timing boundary | Sync ms/call | Async ms/call |
| --- | ---: | ---: |
| CPU `execute_model`, including preparation | 7.55 | 6.46 |
| CPU input preparation | 2.68 | 1.99 |
| **CPU execute, excluding preparation** | **4.87** | **4.46** |
| CUDA-event model range, including preparation | 90.41 | 83.04 |
| CUDA-event preparation, normalized per model call | 0.89 | 0.49 |
| **CUDA-event model range, excluding preparation** | **89.52** | **82.56** |

The earlier `execute_model` number is therefore not GPU execution time. It is a
host-side cumulative function timer around asynchronous CUDA submission. The
GPU remains active after the Python call returns, which explains why the CUDA
event duration is much larger. The original 25.39 ms sync and 30.94 ms async
values from Experiment A are also profiler-inflated CPU wall times; neither is
a direct measurement of kernel execution duration.

![CPU wall timer versus CUDA-event model duration](../results/vllm_cpu_overhead_drilldown_20260809/model_execution_cpu_vs_gpu.png)

## Online versus offline throughput

The offline benchmark reported:

| Mode | Offline output throughput (tok/s) | Offline requests/s |
| --- | ---: | ---: |
| Sync | 810.85 | 16.22 |
| Async | 736.86 | 14.74 |

Comparing these directly with Experiment A would exaggerate frontend overhead:
the instrumented online run achieved only about 310 tok/s because timing,
py-spy, and NSYS ran together.

For a cleaner bound, the later online FlashAttention run omitted CPU timing and
py-spy while retaining NSYS GPU metrics, and used the same model and 300-request
workload:

| Mode | Offline (tok/s) | Online (tok/s) | Online shortfall vs offline |
| --- | ---: | ---: | ---: |
| Sync | 810.85 | 574.15 | 29.2% |
| Async | 736.86 | 607.71 | 17.5% |

This shortfall bounds the combined online-serving cost: HTTP/OpenAI frontend,
streaming output handling, detokenization, serialization, and differences
between the offline and online engine paths. It is not attributable solely to
HTTP.

## Torch-profiler micro-window

The corrected torch profiler recorded five active sync iterations:

- self CPU time: 2.077 s;
- self CUDA time: 1.639 s, or about 328 ms per profiled step;
- `cudaEventSynchronize`: 1.471 s and 70.82% of self CPU time;
- matrix multiplication kernels: 1.230 s and 75.06% of self CUDA time; and
- FlashAttention split-KV kernels: 352.83 ms and 21.53% of self CUDA time.

Most measured CPU time in this micro-window was waiting for the GPU, while CUDA
time was dominated by GEMMs and then attention. The profiler reduced benchmark
throughput to 207 tok/s, so its timing is diagnostic rather than representative.

## Where the time goes

1. **Model-output completion explains the old EngineCore remainder.** Sync
   spends 78.82 ms/step in fallback sampling; async spends 58.99 ms/step
   waiting for the queued model/sample future and 6.73 ms submitting sampling.
2. **Frontend detokenization and token conversion are the clearest CPU
   hotspot.** Detokenizer updates accumulated 19.2-19.5 seconds, while native
   tokenizer iteration and ID conversion led APIServer py-spy self time.
3. **Scheduling and input preparation are profiler-sensitive.** They consumed
   20-24 ms/step in the combined Experiment A run, but the separated follow-up
   measured only 1.25-1.56 ms/step for scheduling and 1.99-2.68 ms/call for
   input preparation. Prefix-cache hashing and NumPy packing remain visible in
   EngineCore samples.
4. **The CPU execute timer is launch time, not GPU duration.** CUDA events
   measured 89.52 ms sync and 82.56 ms async excluding input preparation,
   versus 4.87 ms and 4.46 ms in the follow-up CPU timer.
5. **The serving pipeline leaves substantial GPU gaps.** Roughly 72-74% of
   samples were below 10% GR activity, despite individual bursts reaching full
   activity. The low wall-clock DRAM mean is primarily dilution by idle gaps.

## Optimization candidates

1. **Reduce output-path work:** batch token-ID conversion, avoid detokenizing
   intermediate tokens when the client accepts token IDs, reduce per-token
   allocation, and coalesce streaming/serialization work.
2. **Cache and batch prefix metadata:** reuse block hashes for the shared 10K
   prefix, reduce SHA-256 work and Python/NumPy packing, and move remaining
   per-request hashing or table construction into batched native code.
3. **Overlap CPU phases with GPU execution:** preserve async output processing,
   prepare the next batch while the current batch executes, and extend warmup to
   cover slot-mapping shapes so Triton JIT does not occur during inference.
4. **Optimize the sampling/completion path:** the detailed step regions show
   that synchronous fallback sampling and asynchronous future completion, not
   scheduler bookkeeping, dominate EngineCore wall time in the cleaner run.
5. **Keep profilers separated:** use py-spy at 10-20 Hz without NSYS, collect
   CPU timing independently, and use CUDA events or a server-rooted NSYS launch
   for GPU method duration.

## Limitations

- Every comparison has one repetition and fixed mode order, so no statistical
  significance can be claimed.
- Function timers are cumulative and include nested calls; only the adjusted
  core table avoids known double counting.
- Frontend and EngineCore process totals overlap in wall-clock time.
- py-spy's sampling backlog limits precise self-time attribution.
- The torch-profiler run captures five iterations under heavy profiler overhead.
- Offline versus online is a subsystem bound, not a pure HTTP measurement.
- The follow-up CPU, NSYS, and CUDA-event values come from separate runs; their
  similar 708-741 tok/s throughput makes them comparable, but they are still
  single trials.
- CUDA events measure elapsed work on the instrumented stream. They do not
  replace a multi-stream critical-path analysis.
- NSYS raw CUDA traces are valid, but its built-in NVTX GPU projection was not
  usable for the multiprocess child-worker capture because of timestamp-origin
  mismatch.

## Artifacts

- Instrumented serving: `results/vllm_cpu_overhead_10k_b100_rerun/`
- Offline A/B: `results/vllm_offline_ab_10k_retry/`
- Torch profiler: `results/vllm_torch_profile_10k_retry2/`
- Corrected GPU activity: `results/gpu_busy_analysis_fresh/`
- Detailed step and GPU follow-up:
  `results/vllm_cpu_overhead_drilldown_20260809/`
- Follow-up combined CSV:
  `results/vllm_cpu_overhead_drilldown_20260809/combined_summary.csv`
- Parsed py-spy ranking:
  `results/cpu_overhead_investigation/pyspy_top_self_time.csv`
- Raw NSYS reports remain on `hinton-01` in the corresponding results directory.
