# vLLM Async vs Sync Scheduling Smoke Report

## Objective

Measure how vLLM's asynchronous scheduler changes serving throughput and average DRAM bandwidth
utilization relative to synchronous scheduling while holding the vLLM version, model, workload,
and GPU constant.

## Methodology

- Each scheduling mode started a fresh vLLM server.
- Async and sync used the same random seed (`0`) and identical request workload.
- Each server received 8 warmup prompts before profiling.
- The measured workload used 64 prompts, 1,024 input tokens, 64 requested output tokens,
  unlimited request rate, and maximum concurrency 64.
- Throughput came from the structured output of `vllm bench serve`.
- Nsight Systems sampled GPU metrics at 10 kHz. Server startup and warmup occurred outside the
  profile. Leading and trailing samples with no DRAM activity were removed, while idle gaps
  inside the active request window remained included.
- `dram_avg_pct` is the arithmetic mean across the NSYS DRAM read- and write-bandwidth
  utilization samples. It is a percentage of sustained peak, not GB/s.

## Results

| Metric | Sync | Async | Async - Sync | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Output throughput (tokens/s) | 507.07 | 515.68 | +8.61 | +1.70% |
| Mean DRAM read/write utilization | 7.69% | 7.94% | +0.25 percentage points | +3.27% |
| DRAM p95 utilization | 75.00% | 76.00% | +1.00 percentage point | +1.33% |
| Benchmark duration | 8.078 s | 7.943 s | -0.135 s | -1.67% |

Both trials completed all 64 requests with no failures and generated 4,096 output tokens.

## Interpretation

For this smoke workload, async scheduling was modestly faster: output throughput increased by
1.70%, accompanied by a 3.27% relative increase in the reported mean DRAM utilization metric.
The shorter async benchmark duration is consistent with its higher throughput.

The average DRAM value was only about 8%, while p95 reached 75-76%. This indicates bursty memory
traffic rather than continuously saturated DRAM. The metric also averages separate read and
write percentages; it should not be interpreted as total read-plus-write utilization.

This run used one paired repetition, so the standard deviations are zero by construction and
the result does not establish statistical significance. The observed 1.70% throughput
difference is small enough that run-to-run variance could explain part or all of it.

## Recommended Follow-up

Run at least five paired repetitions, preferably ten, using the same workload. The comparison
runner alternates mode order across repetitions to reduce thermal and ordering bias. A longer,
more decode-heavy workload would also provide a steadier GPU-metric window:

```bash
scripts/vllm_bw.sh compare --remote --detach \
  --host hinton-01 --remote-dir ~/code/attention-bw -- \
  --model phi-3-mini \
  --random-input-len 2048 \
  --random-output-len 128 \
  --num-prompts 256 \
  --warmup-prompts 16 \
  --max-num-seqs 256 \
  --max-concurrency 256 \
  --request-rate inf \
  --repetitions 10
```

## Artifacts

- Manifest: `results/vllm_scheduling_compare_smoke4/manifest.json`
- Trial data: `results/vllm_scheduling_compare_smoke4/trials.csv`
- Mode summary: `results/vllm_scheduling_compare_smoke4/summary.csv`
- Paired comparison: `results/vllm_scheduling_compare_smoke4/paired_comparison.csv`
- Impact summary: `results/vllm_scheduling_compare_smoke4/impact.csv`
- Plot: `results/vllm_scheduling_compare_smoke4/comparison.png`
- Raw NSYS reports and SQLite exports remain on `hinton-01` under
  `~/code/attention-bw/results/vllm_scheduling_compare_smoke4/`.

## 10K Shared-prefix Experiment

### Paper-derived Configuration

The follow-up configuration was taken from *Requests of a Feather Must Flock Together: Batch
Size vs. Prefix Homogeneity in LLM Inference*, read from `Cache_aware_LLM_batching.pdf` using
Ghostscript. The relevant controlled experiments use:

- an NVIDIA RTX 6000 Ada GPU;
- an 8B Llama model and FlashAttention;
- a 10,000-token prefix shared by every request in a homogeneous batch;
- a unique 20-token suffix per request;
- 50 generated tokens per request;
- prefix-cache warmup before measurement; and
- a default maximum batch size of 500.

Two paper-grounded batch sizes were selected: 100, the moderately sized homogeneous batch
highlighted in Experiment 6, and 500, the default maximum and the batch size used in Experiment
1. In each run, the number of prompts, client concurrency, and vLLM `max_num_seqs` were all set
to the selected batch size. Each async/sync pair was repeated three times with alternating
execution order and seeds 0, 1, and 2.

The installed `meta-llama/Meta-Llama-3-8B` checkpoint has an 8,192-token context limit and vLLM
correctly rejected the 10,020-token prompt. Rather than apply an unsafe RoPE override, this test
used the cached `meta-llama/Llama-3.1-8B`, which supports the required context. This is therefore
a paper-derived configuration, not an exact reproduction. Other differences include vLLM
0.22.1 and online-serving output throughput rather than the paper's isolated decode loop.

The vLLM random dataset generated one token-identical 10K prefix for all requests, followed by
independently generated 20-token suffixes. Prefix caching was enabled by vLLM, FlashAttention
was selected through `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, and NSYS GPU metrics were sampled at
2 kHz.

### Results

| Batch size | Metric | Sync mean ± stdev | Async mean ± stdev | Async - Sync | Relative change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Output throughput (tokens/s) | 615.37 ± 15.96 | 571.88 ± 86.75 | -43.48 | -7.07% |
| 100 | Mean DRAM read/write utilization | 3.420% ± 0.154% | 3.360% ± 0.291% | -0.060 points | -1.77% |
| 500 | Output throughput (tokens/s) | 717.27 ± 10.39 | 714.78 ± 32.45 | -2.49 | -0.35% |
| 500 | Mean DRAM read/write utilization | 2.331% ± 0.188% | 2.706% ± 0.142% | +0.374 points | +16.06% |

All 3,600 requests across the 12 measured trials completed successfully with no failures. The
batch-100 trials generated 5,000 output tokens per scheduling mode and repetition; batch 500
generated 25,000.

The paired throughput changes were:

- Batch 100: +0.49%, -22.16%, and +0.20%.
- Batch 500: -5.76%, +2.38%, and +2.38%.

### Interpretation

The batch-100 mean does not show a repeatable async-scheduling penalty. Two paired trials were
effectively tied, while one async trial fell to 472.96 tokens/s and produced the entire negative
mean. Median throughput was 607.71 tokens/s async versus 607.62 tokens/s sync. More repetitions
would be needed to determine why that outlier occurred.

At batch 500, average throughput was effectively unchanged: async was 0.35% slower, well within
the observed run-to-run variation. Two of three paired trials favored async by about 2.38%, while
one favored sync by 5.76%. The current data therefore does not support a throughput advantage
for either scheduling mode on this large homogeneous batch.

Async scheduling increased the batch-500 DRAM metric by 0.374 percentage points, or 16.06%
relative, without increasing average throughput. Because absolute mean utilization remained
below 3% and the metric averages separate NSYS read and write percentages, the relative change
should not be interpreted as a 16% increase in total physical bandwidth. DRAM p95 was also
bursty: its mean across trials was 13.25% async and 10.67% sync at batch 500.

Increasing the homogeneous batch from 100 to 500 raised throughput to roughly 715-717 tokens/s,
consistent with the paper's observation that throughput improves with batch size before
plateauing. Absolute values should not be compared directly with the paper because the model
checkpoint, vLLM version, and throughput measurement boundary differ.

### 10K Experiment Artifacts

- Batch 100: `results/vllm_scheduling_10k_b100_llama31/`
- Batch 500: `results/vllm_scheduling_10k_b500_llama31/`
- Each directory contains `manifest.json`, `trials.csv`, `summary.csv`,
  `paired_comparison.csv`, `impact.csv`, and `comparison.png`.
- Raw NSYS reports and SQLite exports remain on `hinton-01` in the corresponding result
  directories under `~/code/attention-bw/results/`.

## Multi-wave 10K Scheduler Follow-up

### Motivation and Configuration

The preceding 10K experiment submitted exactly as many prompts as the configured concurrency.
It therefore created one request wave: async scheduling still ran at each decode iteration, but
there were no queued replacement requests to admit as active requests completed. That experiment
cannot determine how async scheduling behaves under sustained queue turnover.

The follow-up retained the same Llama 3.1 8B, 10K shared prefix, 20-token unique suffix,
50-token output, FlashAttention, unlimited request rate, and three paired repetitions. Only the
number of prompts and NSYS sampling frequency changed:

| Configured batch | Prompts | Planned request waves | NSYS frequency |
| ---: | ---: | ---: | ---: |
| 100 | 1,000 | 10 | 500 Hz |
| 500 | 2,500 | 5 | 500 Hz |

The comparison runner alternated async/sync order and restarted the server for every trial.
Across both configurations, all 21,000 measured requests completed without failures.

### Multi-wave Results

| Batch size | Metric | Sync mean ± stdev | Async mean ± stdev | Async - Sync | Relative change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Output throughput (tokens/s) | 645.83 ± 19.58 | 631.24 ± 10.37 | -14.59 | -2.26% |
| 100 | Mean DRAM read/write utilization | 5.091% ± 0.116% | 5.122% ± 0.137% | +0.031 points | +0.61% |
| 500 | Output throughput (tokens/s) | 567.11 ± 26.57 | 578.81 ± 26.31 | +11.71 | +2.06% |
| 500 | Mean DRAM read/write utilization | 1.951% ± 0.025% | 2.000% ± 0.084% | +0.048 points | +2.47% |

The paired throughput changes were:

- Batch 100: +0.76%, -5.84%, and -1.53%.
- Batch 500: -0.80%, -5.28%, and +13.17%.

### Updated Interpretation

Adding queued request waves changes the conclusion from “the workload did not exercise
scheduling” to “the measured async effect is small and inconsistent.” Async scheduling was
2.26% slower on average at concurrency 100 and 2.06% faster at concurrency 500. Neither result
was consistent across all three pairs, and the batch-500 paired throughput delta had much
greater variance than its mean.

These data do not support the claim that async scheduling does nothing for large contexts. At
concurrency 500 it produced a small positive mean throughput change under real queue turnover.
They also do not establish a reliable async advantage: one pair favored async strongly, while
the other two favored sync. More repetitions are required to separate a roughly 2% effect from
run-to-run noise.

The small effect is plausible for this workload. Every decode step performs attention over a
10K context, so GPU work per scheduling decision is large and CPU scheduling overhead is
already heavily amortized. Async scheduling should matter more when CPU scheduling gaps are a
larger fraction of each iteration, such as with smaller models, shorter contexts, mixed output
lengths, or more frequent request completions.

DRAM utilization moved in the positive direction for async in both aggregate comparisons, but
the absolute differences were only 0.031 and 0.048 percentage points. The relative percentages
look larger because the reported read/write-average baseline is low. They are not evidence of a
material increase in physical DRAM bandwidth.

The batch-500 multi-wave throughput is lower than its one-wave result because online-serving
throughput includes repeated request admission and the unique-suffix prefill work for subsequent
waves. The one-wave result mostly measured one warmed, stable active set.

### Multi-wave Artifacts

- Batch 100: `results/vllm_scheduling_10k_multiwave_b100/`
- Batch 500: `results/vllm_scheduling_10k_multiwave_b500/`
- Each directory contains `manifest.json`, `trials.csv`, `summary.csv`,
  `paired_comparison.csv`, `impact.csv`, and `comparison.png`.
- Raw NSYS reports and SQLite exports remain on `hinton-01` in the corresponding directories.
