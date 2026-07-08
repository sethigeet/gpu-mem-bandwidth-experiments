# Component Bandwidth 10K Shared-prefix Report

## Methodology

- Workload: Phi-3-mini-shaped synthetic decode ladder with `prefix_len=10000`, `dtype=fp16`, `layout=shared`, and `batch_size=auto`.
- Throughput source: `results/component_bw_nsys_10k_metrics.csv` from the full component matrix run. This measures wall-clock decode throughput with CUDA synchronization around each measured stage.
- Bandwidth source: per-stage Nsight Compute CSVs matched by `results/component_bw_ncu_10k_*.csv`. NCU profiles one measured decode token per stage with warmup excluded through `component_bw:<stage>:iter` NVTX filtering.
- Counters: `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, and `gpu__time_duration.sum`.
- Aggregation: effective GB/s is total read+write bytes divided by total profiled kernel duration. DRAM/SM percentages are duration-weighted averages across kernels in each stage.
- Caveat: NCU replays kernels to collect counters, so its duration and bandwidth are reliable for hardware-counter attribution but are not the same measurement as wall-clock throughput.

## Results

![Component bandwidth summary](component_bw_10k_report.png)

### Stage Summary

| stage            | batch_size | throughput_toks_s | per_token_ms | ncu_bandwidth_gb_s | ncu_dram_pct_weighted | ncu_sm_pct_weighted | kernel_count |
| ---------------- | ---------- | ----------------- | ------------ | ------------------ | --------------------- | ------------------- | ------------ |
| attention_kernel | 256        | 9655.60           | 0.10         | 7.75               | 0.85                  | 86.49               | 1            |
| attention_layer  | 256        | 10555.30          | 0.09         | 12.55              | 1.38                  | 85.92               | 5            |
| mlp              | 256        | 935397.28         | 0.00         | 541.51             | 59.48                 | 48.05               | 5            |
| block            | 256        | 10097.80          | 0.10         | 24.36              | 2.68                  | 84.79               | 28           |
| blocks           | 256        | 288.21            | 3.47         | 23.35              | 2.56                  | 84.86               | 896          |
| model            | 256        | 276.39            | 3.62         | 23.54              | 2.59                  | 84.83               | 907          |
| paged_attention  | 69         | 914.78            | 1.09         | 833.77             | 91.50                 | 35.21               | 21           |
| paged_model      | 44         | 28.20             | 35.47        | 828.06             | 90.88                 | 34.38               | 1195         |

### NCU Kernel-type Time Breakdown

| stage            | kernel_type  | kernel_count | ncu_duration_ms | ncu_bandwidth_gb_s | ncu_dram_pct_weighted | ncu_sm_pct_weighted |
| ---------------- | ------------ | ------------ | --------------- | ------------------ | --------------------- | ------------------- |
| attention_kernel | attention    | 1            | 16.54           | 7.75               | 0.85                  | 86.49               |
| attention_layer  | linear       | 4            | 0.18            | 459.08             | 50.43                 | 31.63               |
| attention_layer  | attention    | 1            | 16.55           | 7.74               | 0.85                  | 86.51               |
| mlp              | linear       | 3            | 0.30            | 532.75             | 58.50                 | 50.52               |
| mlp              | activation   | 2            | 0.02            | 682.55             | 75.29                 | 8.26                |
| block            | linear       | 7            | 0.48            | 500.52             | 54.97                 | 44.05               |
| block            | other        | 2            | 0.02            | 307.81             | 33.88                 | 0.76                |
| block            | activation   | 18           | 0.09            | 495.37             | 54.80                 | 8.01                |
| block            | attention    | 1            | 16.61           | 7.71               | 0.85                  | 86.49               |
| blocks           | activation   | 576          | 2.88            | 492.13             | 54.42                 | 7.89                |
| blocks           | linear       | 224          | 15.47           | 496.82             | 54.56                 | 43.95               |
| blocks           | other        | 64           | 0.65            | 312.54             | 34.45                 | 0.77                |
| blocks           | attention    | 32           | 555.28          | 7.39               | 0.81                  | 86.50               |
| model            | paged_gather | 1            | 0.00            | 377.88             | 41.89                 | 3.51                |
| model            | activation   | 583          | 2.91            | 491.71             | 54.37                 | 7.91                |
| model            | other        | 66           | 0.69            | 325.26             | 35.85                 | 1.64                |
| model            | linear       | 225          | 15.84           | 498.88             | 54.79                 | 44.71               |
| model            | attention    | 32           | 560.82          | 7.32               | 0.80                  | 86.47               |
| paged_attention  | paged_gather | 8            | 20.28           | 831.64             | 91.27                 | 48.58               |
| paged_attention  | activation   | 8            | 20.74           | 811.53             | 89.06                 | 23.88               |
| paged_attention  | linear       | 4            | 0.11            | 694.10             | 76.32                 | 20.16               |
| paged_attention  | attention    | 1            | 9.71            | 887.35             | 97.38                 | 31.67               |
| paged_model      | paged_gather | 129          | 427.49          | 833.57             | 91.48                 | 47.82               |
| paged_model      | activation   | 711          | 439.73          | 809.83             | 88.88                 | 23.69               |
| paged_model      | other        | 66           | 0.67            | 63.89              | 7.04                  | 0.30                |
| paged_model      | linear       | 257          | 9.97            | 759.84             | 83.48                 | 13.96               |
| paged_model      | attention    | 32           | 210.51          | 860.65             | 94.46                 | 30.51               |

## Interpretation

- The biggest throughput drop occurs when moving from one block to all 32 blocks, which is expected because the stage multiplies attention and MLP work by layer count.
- The paged model stage is much slower than the dense shared full model in this PyTorch approximation because page gathering materializes dense K/V tensors and adds substantial indexing/copy overhead.
- Use the NCU DRAM/SM columns above as the reliable bandwidth/utilization source for this run. The earlier NSYS GPU-metric timeline failed on this host with a GPU metric ordering error and was not used for bandwidth conclusions.

## Artifacts

- Throughput CSV: `results/component_bw_nsys_10k_metrics.csv`
- NCU stage summary CSV: `results/component_bw_10k_report_ncu_summary.csv`
- NCU kernel-type summary CSV: `results/component_bw_10k_report_ncu_kernel_types.csv`
- Plot: `results/component_bw_10k_report.png`

## Fixed Batch 32 Comparison

To make `model` and `paged_model` throughput comparable at the same batch size, I also ran the full 10K shared-prefix ladder with `--batch-size 32`. This keeps the workload in VRAM and removes the auto-batch-size difference from the throughput comparison.

![Fixed batch 32 component bandwidth summary](component_bw_10k_b32_report.png)

### Fixed Batch 32 Stage Summary

| stage            | batch_size | throughput_toks_s | per_token_ms | ncu_bandwidth_gb_s | ncu_dram_pct_weighted | ncu_sm_pct_weighted | kernel_count |
| ---------------- | ---------- | ----------------- | ------------ | ------------------ | --------------------- | ------------------- | ------------ |
| attention_kernel | 32         | 10040.87          | 0.10         | 56.24              | 6.17                  | 77.66               | 64.00        |
| attention_layer  | 32         | 11015.73          | 0.09         | 89.73              | 9.86                  | 74.23               | 576.00       |
| mlp              | 32         | 162015.06         | 0.01         | 787.28             | 86.49                 | 7.48                | 384.00       |
| block            | 32         | 9614.84           | 0.10         | 140.99             | 15.49                 | 67.60               | 2112.00      |
| blocks           | 32         | 272.68            | 3.67         | 139.92             | 15.37                 | 67.63               | 67584.00     |
| model            | 32         | 268.86            | 3.72         | 141.95             | 15.60                 | 67.44               | 68288.00     |
| paged_attention  | 32         | 1103.16           | 0.91         | 836.32             | 91.79                 | 34.77               | 832.00       |
| paged_model      | 32         | 31.73             | 31.51        |                    |                       |                     |              |

The fixed-batch throughput comparison is the clean apples-to-apples result:

- `model` at batch 32: `268.86 tok/s`
- `paged_model` at batch 32: `31.73 tok/s`
- `paged_model` is about `8.5x` slower at the same batch size.

NCU counters are available through `paged_attention`. The fixed-batch `paged_model` NCU stage was stopped after more than 15 hours of active GPU execution, so it is intentionally omitted from the fixed-batch NCU summary. The earlier auto-batch `paged_model` NCU result remains useful as contextual evidence that the PyTorch paged-KV approximation is highly memory-bandwidth bound.

Fixed-batch artifacts:

- Report: `results/component_bw_10k_b32_report.md`
- Throughput CSV: `results/component_bw_10k_b32_throughput.csv`
- NCU stage summary CSV: `results/component_bw_10k_b32_report_ncu_summary.csv`
- NCU kernel-type summary CSV: `results/component_bw_10k_b32_report_ncu_kernel_types.csv`
- Plot: `results/component_bw_10k_b32_report.png`
