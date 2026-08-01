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

1. **Frontend detokenization and token conversion are the clearest CPU
   hotspot.** Detokenizer updates accumulated 19.2-19.5 seconds, while native
   tokenizer iteration and ID conversion led APIServer py-spy self time.
2. **Scheduling and input preparation consume about one quarter of each core
   step.** Together they used 23.69 ms sync and 19.86 ms async per step.
   Prefix-cache hashing and NumPy packing are visible in EngineCore samples.
3. **The serving pipeline leaves substantial GPU gaps.** Roughly 72-74% of
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
4. **Separate profilers in follow-up runs:** use py-spy at 10-20 Hz without
   NSYS, and collect timing hooks/NSYS in a separate run. This will produce a
   representative throughput baseline and avoid the observed sampling backlog.

## Limitations

- Every comparison has one repetition and fixed mode order, so no statistical
  significance can be claimed.
- Function timers are cumulative and include nested calls; only the adjusted
  core table avoids known double counting.
- Frontend and EngineCore process totals overlap in wall-clock time.
- py-spy's sampling backlog limits precise self-time attribution.
- The torch-profiler run captures five iterations under heavy profiler overhead.
- Offline versus online is a subsystem bound, not a pure HTTP measurement.

## Artifacts

- Instrumented serving: `results/vllm_cpu_overhead_10k_b100_rerun/`
- Offline A/B: `results/vllm_offline_ab_10k_retry/`
- Torch profiler: `results/vllm_torch_profile_10k_retry2/`
- Corrected GPU activity: `results/gpu_busy_analysis_fresh/`
- Parsed py-spy ranking:
  `results/cpu_overhead_investigation/pyspy_top_self_time.csv`
- Raw NSYS reports remain on `hinton-01` in the corresponding results directory.
