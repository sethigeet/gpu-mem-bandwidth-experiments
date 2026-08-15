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
- one authoritative timing-only async trial and one sync trial
- separate profiler-isolation trials with NSYS at 500 Hz and/or a 60-second
  py-spy subprocess capture with native frames

The primary CPU measurements below come from the latest timing-only rerun.
Profiler-enabled trials are retained only to quantify measurement distortion;
their timings are not used as production estimates.

## Experiment status

| Experiment                    | Status                        | Result                                                                                                                       |
| ----------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Authoritative CPU drilldown   | Success                       | Latest timing-only sync/async pair measured every exclusive EngineCore region without py-spy or NSYS                         |
| Profiler-isolation validation | Success                       | Timing-only, py-spy-only, NSYS-only, and py-spy+NSYS pairs identified py-spy as the dominant source of timing distortion     |
| GPU timing                    | Success after NSYS workaround | NSYS captured 17K+ kernels per mode; CUDA events supplied per-call GPU duration after NSYS's child-process projection failed |
| Online versus offline         | Success after retry           | Initial FlashInfer sampler JIT failed; rerun with `VLLM_USE_FLASHINFER_SAMPLER=0` completed                                  |
| Torch-profiler window         | Success after retry           | The old profiler environment variable produced no trace; `--profiler-config` generated a 30.8 MB trace and summary           |
| Fresh GPU-busy analysis       | Success after correction      | Two fresh NSYS traces analyzed; GR-active extraction was corrected to exclude cycle-count metrics                            |

## Authoritative timing-only serving results

| Mode  | Output throughput (tok/s) | Duration (s) | Mean TTFT (ms) | Mean TPOT (ms) |
| ----- | ------------------------: | -----------: | -------------: | -------------: |
| Sync  |                    694.60 |        21.60 |       2,018.20 |         100.62 |
| Async |                    710.32 |        21.12 |       2,007.75 |          98.19 |

Async throughput was 2.26% higher, with slightly lower TTFT and TPOT. These are
the serving results used for the CPU phase budget below.

## Per-iteration EngineCore budget

Exclusive timing regions inside `EngineCore.step` and
`EngineCore.step_with_batch_queue` resolve the code previously hidden by
subtraction-based accounting.

| Exclusive EngineCore phase                          | Sync ms/step |    Sync % | Async ms/step |   Async % |
| --------------------------------------------------- | -----------: | --------: | ------------: | --------: |
| Scheduler                                           |         1.83 |      1.79 |          2.23 |      2.27 |
| Execute-model submission                            |         7.32 |      7.17 |          8.23 |      8.37 |
| Sync `sample_tokens` (actual GPU work)              |        92.98 |     91.01 |             — |         — |
| Async initial sample submission                     |            — |         — |          6.76 |      6.88 |
| Async model/sample future wait                      |            — |         — |         82.50 |     83.96 |
| Update from output                                  |         0.42 |      0.41 |          0.50 |      0.51 |
| Request checks, grammar, queues, and abort handling |         0.01 |      0.01 |          0.01 |      0.01 |
| Timer reconciliation error                          |        -0.39 |     -0.38 |         -1.97 |     -2.01 |
| **EngineCore step**                                 |   **102.16** | **100.0** |     **98.26** | **100.0** |

The sync remainder is overwhelmingly the fallback call to
`model_executor.sample_tokens()` after `execute_model()` returns `None`. In the
async batch-queue path, the corresponding cost appears mainly in
`future.result()`, with smaller launch-side costs for model and sample
submission. The small negative reconciliation row is timing/context-manager
instrumentation overhead, not another execution phase.

The sync path spends 91% of its step in fallback sampling. The async path
spends 84% waiting for the queued model/sample future and another 6.9%
submitting sampling. Scheduler bookkeeping is approximately 2% in both modes.

A profiler-isolation check validated the choice of the timing-only result.
NSYS-only scheduler timing stayed close at 2.11 ms sync and 1.91 ms async.
Attaching py-spy increased the apparent scheduler time by 4.0-4.8x, reduced
throughput by 56-57%, fragmented 200 iterations into 400-500 calls, and fell
20-28 seconds behind. All py-spy timing values are therefore excluded from the
reported CPU budget rather than presented as an alternative measurement.

Frontend output processing is in a separate process and can overlap EngineCore,
so it is not additive with the table:

| Frontend function                                    |                 Sync |                Async |
| ---------------------------------------------------- | -------------------: | -------------------: |
| `OutputProcessor.process_outputs` per call           | 15.59 ms (240 calls) | 14.95 ms (239 calls) |
| Output-processing total                              |               3.74 s |               3.57 s |
| `BaseIncrementalDetokenizer.update` per token update |              0.21 ms |              0.20 ms |
| Detokenization total, 15,400 calls                   |               3.27 s |               3.09 s |

Detokenization consumed almost as much cumulative time as the complete output
processor. The totals are cumulative function times and may overlap across
threads/processes; they should not be summed with EngineCore wall time.

## GPU activity and idle time

The corrected fresh NSYS analysis used `GR Active [Throughput %]` as the busy
proxy and excluded `GR Active [Cycles Active]`.

| Mode  | GR active mean % | SM active mean % | Median % | Samples below 10% | Samples above 80% | DRAM mean % |
| ----- | ---------------: | ---------------: | -------: | ----------------: | ----------------: | ----------: |
| Sync  |            24.44 |            22.11 |        0 |            74.21% |            23.37% |        5.78 |
| Async |            27.54 |            24.58 |        0 |            71.70% |            26.95% |        5.67 |

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

NSYS therefore confirmed that the workload executed and supplied kernel-level
traces and summaries, but it did **not** independently confirm the duration of
the model method. The GPU method-duration numbers below are CUDA-event results,
not NSYS NVTX-projection results.

| Timing boundary                                          | Sync ms/call | Async ms/call |
| -------------------------------------------------------- | -----------: | ------------: |
| Authoritative CPU `execute_model`, including preparation |         7.25 |          8.16 |
| Authoritative CPU input preparation                      |         1.55 |          2.17 |
| **Authoritative CPU execute, excluding preparation**     |     **5.70** |      **5.99** |
| CUDA-event model range, including preparation            |        90.41 |         83.04 |
| CUDA-event preparation, normalized per model call        |         0.89 |          0.49 |
| **CUDA-event model range, excluding preparation**        |    **89.52** |     **82.56** |

The CPU `execute_model` number is therefore not GPU execution time. It is a
host-side cumulative function timer around asynchronous CUDA submission. The
GPU remains active after the Python call returns, which explains why the CUDA
event duration is much larger. The CPU and CUDA-event values come from separate
clean runs because adding CUDA events changes synchronization behavior; neither
CPU launch timer is a direct measurement of kernel execution duration.

The CUDA-event model duration does match the complete CPU-observed EngineCore
step in the same event-instrumented run:

| Mode  | CUDA-event model range, including preparation | CPU EngineCore step | Absolute difference | Difference versus step |
| ----- | --------------------------------------------: | ------------------: | ------------------: | ---------------------: |
| Sync  |                                      90.41 ms |            94.98 ms |             4.57 ms |                   4.8% |
| Async |                                      83.04 ms |            83.43 ms |             0.39 ms |                   0.5% |

This agreement resolves the apparent CPU/GPU mismatch. The CPU
`execute_model` wrapper measures only asynchronous launch work. Sync then waits
mainly in fallback `sample_tokens()`, while async waits mainly in
`future.result()`. Once those completion waits are included, the full CPU step
tracks GPU elapsed time closely. A direct EngineCore-worker NSYS capture is
still required for an independent NSYS range-duration validation.

## Online versus offline throughput

The offline benchmark reported:

| Mode  | Offline output throughput (tok/s) | Offline requests/s |
| ----- | --------------------------------: | -----------------: |
| Sync  |                            810.85 |              16.22 |
| Async |                            736.86 |              14.74 |

The online FlashAttention run omitted CPU timing and py-spy while retaining
NSYS GPU metrics, and used the same model and 300-request workload:

| Mode  | Offline (tok/s) | Online (tok/s) | Online shortfall vs offline |
| ----- | --------------: | -------------: | --------------------------: |
| Sync  |          810.85 |         574.15 |                       29.2% |
| Async |          736.86 |         607.71 |                       17.5% |

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

1. **Model-output completion dominates EngineCore time.** Sync spends 92.98
   ms/step in fallback sampling; async spends 82.50 ms/step waiting for the
   queued model/sample future and 6.76 ms submitting sampling.
2. **Frontend output processing remains material but overlaps EngineCore.** It
   accumulated 3.57-3.74 seconds, of which detokenizer updates accumulated
   3.09-3.27 seconds.
3. **Scheduler bookkeeping is small in the authoritative run.** It measured
   1.83 ms sync and 2.23 ms async, or approximately 2% of EngineCore time.
   Profiler isolation showed that NSYS alone remains near this result, whereas
   py-spy makes wall-clock function timings unusable.
4. **The CPU execute timer is launch time, not GPU duration.** CUDA events
   measured 89.52 ms sync and 82.56 ms async excluding input preparation,
   versus 5.70 ms and 5.99 ms in the authoritative CPU timer. In the same
   event-instrumented run, the GPU range was within 4.8% sync and 0.5% async of
   the complete EngineCore step.
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
   scheduler bookkeeping, dominate EngineCore wall time in the authoritative
   run.
5. **Keep profilers separated:** collect CPU timing independently. If py-spy is
   needed for qualitative hotspot discovery, use a lower 10-20 Hz rate in a
   separate run. Use CUDA events or a server-rooted NSYS launch for GPU method
   duration.
6. **Reset and summarize timers after warmup:** report median and p95 in
   addition to mean, and repeat timing-only trials before assigning a precise
   production scheduler cost.

## Limitations

- Every comparison has one repetition and fixed mode order, so no statistical
  significance can be claimed.
- Function timers are cumulative and include nested calls; only the adjusted
  core table avoids known double counting.
- Frontend and EngineCore process totals overlap in wall-clock time.
- py-spy's sampling backlog limits precise self-time attribution.
- The torch-profiler run captures five iterations under heavy profiler overhead.
- Offline versus online is a subsystem bound, not a pure HTTP measurement.
- The authoritative CPU and CUDA-event values come from separate runs. Their
  serving throughputs are in the same general range, but they are still single
  trials and CUDA-event instrumentation changes synchronization behavior.
- The profiler-isolation matrix also has one trial per condition. It establishes
  the source of the large profiler-dependent discrepancy but not a tight
  confidence interval for clean timings.
- Cumulative timing begins before the eight-request warmup, and only aggregate
  mean/min/max are retained. Warmup variation and a few expensive calls can
  move the reported mean.
- CUDA events measure elapsed work on the instrumented stream. They do not
  replace a multi-stream critical-path analysis.
- NSYS raw CUDA traces are valid, but its built-in NVTX GPU projection was not
  usable for the multiprocess child-worker capture because of timestamp-origin
  mismatch. Consequently, the reported GPU method duration is CUDA-event
  validated, not independently NSYS-range validated.

## Artifacts

- Offline A/B: `results/vllm_offline_ab_10k_retry/`
- Torch profiler: `results/vllm_torch_profile_10k_retry2/`
- Corrected GPU activity: `results/gpu_busy_analysis_fresh/`
- GPU timing follow-up and same-run CUDA-event step data:
  `results/vllm_cpu_overhead_drilldown_20260809/`
- Authoritative CPU timing and profiler-isolation validation:
  `results/vllm_scheduler_profiler_matrix_20260811/`
- Consolidated profiler-matrix CSV:
  `results/vllm_scheduler_profiler_matrix_20260811/profiler_matrix_summary.csv`
- Raw NSYS reports remain on `hinton-01` in the corresponding results directory.
