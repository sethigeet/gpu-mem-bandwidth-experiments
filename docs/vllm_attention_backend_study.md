# vLLM Attention Kernel Bandwidth and Throughput Comparison

## Objective

Compare end-to-end serving throughput, latency, and DRAM bandwidth utilization for
vLLM attention backends across:

- a short-prompt workload and a 10K-token shared-prefix workload;
- unlimited and Poisson request arrivals; and
- synchronous and asynchronous scheduling.

The tested backends were `FLASH_ATTN`, `FLASHINFER`, and `TRITON_ATTN`.

## Methodology

All runs used vLLM 0.22.1 on an NVIDIA RTX 6000 Ada Generation GPU with driver
595.84. FCFS scheduling, prefix caching, float16 model execution, and
`VLLM_USE_FLASHINFER_SAMPLER=0` were held constant. Each configuration had one
paired async/sync repetition with seed 0. Async ran first in every pair.

Each backend/workload pair was run through `scripts/vllm-server.sh compare`, with
`VLLM_ATTENTION_BACKEND` selecting the backend. Nsight Systems sampled GPU metrics at 2 kHz
during the measured `vllm bench serve` window. The reported
DRAM mean is the arithmetic mean of the separate read- and write-bandwidth
utilization samples after trimming inactive leading and trailing edges. It is a
percentage of sustained peak, not read-plus-write GB/s.

### Workloads

| Workload | Model | Prompt | Output | Requests | Concurrency | Request rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Short, infinite | Phi-3 Mini 4K | 1,024 random | 64 | 64 | 64 | unlimited |
| Short, Poisson | Phi-3 Mini 4K | 1,024 random | 64 | 64 | 64 | 5 requests/s |
| Long prefix, infinite | Llama 3.1 8B | 10,000 shared + 20 unique | 50 | 300 | 100 | unlimited |
| Long prefix, Poisson | Llama 3.1 8B | 10,000 shared + 20 unique | 50 | 300 | 100 | 5 requests/s |

The finite-rate client used vLLM's default burstiness of 1.0, producing
exponentially distributed inter-arrival times.

## Results

### Short prompts, unlimited arrivals

| Backend | Mode | Throughput (tok/s) | DRAM mean % | DRAM p95 % | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FlashAttention | Sync | 484.75 | 7.44 | 78 | 3,180.57 | 78.25 |
| FlashAttention | Async | **500.28** | 8.39 | 82 | **2,962.14** | **77.66** |
| Triton Attention | Sync | 467.13 | 7.21 | 56 | 3,225.35 | 82.47 |
| Triton Attention | Async | 473.26 | 7.63 | 78 | 3,237.19 | 80.83 |

FlashAttention delivered 3.8% more sync throughput and 5.7% more async
throughput than Triton Attention. Its mean DRAM utilization was slightly higher,
especially in async mode. FlashInfer could not run this model because Phi-3's
attention head size is unsupported by the installed FlashInfer backend.

### Long shared prefix, unlimited arrivals

| Backend | Mode | Throughput (tok/s) | DRAM mean % | DRAM p95 % | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FlashAttention | Sync | 574.15 | 4.28 | 31 | 2,542.42 | 118.56 |
| FlashAttention | Async | 607.71 | 4.21 | 30 | **2,116.89** | 119.18 |
| FlashInfer | Sync | **780.47** | **5.72** | **50** | 3,037.71 | **60.55** |
| FlashInfer | Async | **763.35** | **5.75** | **49** | 3,022.00 | **62.16** |
| Triton Attention | Sync | 730.64 | 5.25 | 43 | **2,505.37** | 80.90 |
| Triton Attention | Async | 740.81 | 5.61 | 48 | 2,996.03 | 68.67 |

FlashInfer had the highest saturated throughput: 35.9% above FlashAttention in
sync mode and 25.6% above it in async mode. Triton was second, 27.3% and 21.9%
above FlashAttention respectively. FlashInfer also used the most mean and p95
DRAM bandwidth, but the additional traffic coincided with substantially higher
throughput and roughly half FlashAttention's TPOT.

### Short prompts, Poisson arrivals

| Backend | Mode | Throughput (tok/s) | DRAM mean % | DRAM p95 % | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FlashAttention | Sync | 289.15 | 15.36 | 90 | **177.58** | 22.52 |
| FlashAttention | Async | 290.06 | 16.40 | 89.95 | **179.18** | **20.40** |
| Triton Attention | Sync | 287.42 | 15.03 | 90 | 199.93 | 23.01 |
| Triton Attention | Async | 288.56 | 16.07 | 90 | 193.39 | 21.89 |

Throughput was controlled by the offered request rate, so the backend difference
was less than 1%. FlashAttention had 11.2% lower sync TTFT and 7.3% lower async
TTFT than Triton, with modestly lower TPOT.

### Long shared prefix, Poisson arrivals

| Backend | Mode | Throughput (tok/s) | DRAM mean % | DRAM p95 % | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FlashAttention | Sync | 245.42 | 23.06 | 90 | 174.26 | 28.55 |
| FlashAttention | Async | 245.52 | 24.14 | 91 | 194.49 | 26.96 |
| FlashInfer | Sync | 245.45 | **28.17** | **93** | **153.30** | **22.71** |
| FlashInfer | Async | 245.55 | **29.63** | **93** | **167.03** | **20.91** |
| Triton Attention | Sync | 245.46 | 25.15 | 92 | 172.77 | 24.94 |
| Triton Attention | Async | 245.50 | 27.18 | **93** | 189.96 | 23.41 |

All backends produced the same rate-limited throughput. FlashInfer had the best
latency: relative to FlashAttention, its TTFT was 12.0% lower sync and 14.1%
lower async, while TPOT was 20.5% and 22.4% lower. Triton was between the two.

The Poisson runs had higher mean and p95 DRAM percentages than the saturated
large-batch runs despite lower throughput. At low arrival rates, requests execute
in smaller batches and generate short, intense memory bursts. Therefore these
wall-clock percentages should not be interpreted as bytes per generated token
or kernel efficiency.

## Conclusions

1. **FlashInfer was fastest for the 10K shared-prefix workload.** It reached
   780 tok/s sync and 763 tok/s async, compared with 574 and 608 tok/s for
   FlashAttention.
2. **Triton was competitive on the long-prefix workload.** It achieved
   731-741 tok/s while using slightly less mean DRAM bandwidth than FlashInfer.
3. **FlashAttention was best for the tested short workload.** It was 4-6% faster
   than Triton under unlimited arrivals and had better latency at 5 requests/s.
4. **Kernel choice depends on model compatibility.** FlashInfer rejected
   Phi-3 Mini because its head size is unsupported; this is a hard compatibility
   constraint rather than a benchmark failure.
5. **Poisson throughput is not a kernel-speed metric.** At 5 requests/s all
   compatible kernels were arrival-limited. TTFT, TPOT, and DRAM burst behavior
   are the useful comparisons in those runs.

## Limitations

- There was only one paired repetition. Differences may include run-to-run,
  thermal, and fixed execution-order effects; no significance claim is possible.
- Async always ran before sync, and backend order was FlashAttention,
  FlashInfer, then Triton.
- The short and long workloads use different models, so results compare
  backends within each workload, not model families.
- FlashInfer has no short-workload result because the canonical Phi-3 model is
  incompatible. A follow-up using a shared compatible model would be required
  for a complete three-backend short-context comparison.
- Mean DRAM utilization includes internal idle gaps and averages read and write
  percentages. It is not equivalent to kernel-only Nsight Compute utilization.

## Artifacts

- Fetched summaries and logs:
  `results/vllm_attention_kernel_matrix/`
- Per-configuration inputs: each result directory's `manifest.json`
- Per-mode metrics: each result directory's `trials.csv` and `summary.csv`
- Run status: `results/vllm_attention_kernel_matrix/status.tsv`
- Raw `.nsys-rep` and `.sqlite` traces remain in the remote run directory.
