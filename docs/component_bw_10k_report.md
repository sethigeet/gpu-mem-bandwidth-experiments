# Component Bandwidth 10K Shared-prefix Batch Sweep

## Methodology

- Hardware: NVIDIA RTX 6000 Ada Generation (48 GB).
- Workload: Phi-3-mini-shaped synthetic decode ladder with `prefix_len=10000`, `dtype=fp16`, and `layout=shared`.
- Batch sizes: `1, 2, 4, 8, 16, 32, 40, 48, 64, 128, 256, 512, 1024`, subject to each stage's memory limit.
- Throughput: 64 measured decode iterations after five warmup iterations, with CUDA synchronization around each stage.
- DRAM utilization: Nsight Compute profiles one measured decode iteration per stage and batch size. Warmup is excluded by the `component_bw:<stage>:iter` NVTX filter.
- NCU counters: `dram__throughput.avg.pct_of_peak_sustained_elapsed` and `gpu__time_duration.sum`.
- Aggregation: reported DRAM utilization is the kernel-duration-weighted mean across the measured stage.
- Saturation batch: the first measured batch that reaches at least 95% of that stage's maximum observed throughput.

The paged approximation gathers 10K-token K/V blocks into batch-sized dense tensors. Its memory use therefore grows much faster with batch size than the shared dense-cache stages. Infeasible points are recorded as explicit skips instead of aborting the sweep.

## Component Parameter Counts

These are the trainable tensor parameters instantiated by each synthetic stage. They exclude the K/V cache and temporary activations, which are runtime data rather than model parameters. All projections are bias-free. Parameter storage assumes the benchmark's FP16 dtype.

| stage | included modules | parameters | FP16 parameter storage |
| --- | --- | ---: | ---: |
| attention_kernel | direct SDPA only | 0 | 0 B |
| attention_layer | Q, K, V, and output projections | 37,748,736 (37.75M) | 72.00 MiB |
| mlp | gate, up, and down projections | 75,497,472 (75.50M) | 144.00 MiB |
| block | attention + MLP + two RMSNorms | 113,252,352 (113.25M) | 216.01 MiB |
| blocks | 32 decoder blocks | 3,624,075,264 (3.624B) | 6.75 GiB |
| model | embeddings + 32 blocks + final norm + LM head | 3,821,079,552 (3.821B) | 7.12 GiB |
| paged_attention | same parameterized attention layer; paged K/V is runtime state | 37,748,736 (37.75M) | 72.00 MiB |
| paged_model | same model parameters; paged K/V is runtime state | 3,821,079,552 (3.821B) | 7.12 GiB |

## Results

![Component throughput and DRAM utilization versus batch size](../results/component_bw_10k_batch_sweep_report.png)

### Saturation Summary

| stage | saturation batch (95%) | peak batch | peak throughput (tok/s) | DRAM util at peak | largest feasible batch | endpoint throughput (tok/s) | endpoint DRAM util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention_kernel | 2 | 2 | 14,119.90 | 93.15% | 1024 | 9,069.98 | 0.21% |
| attention_layer | 4 | 40 | 10,539.44 | 7.40% | 1024 | 9,447.82 | 0.30% |
| mlp | 512 | 512 | 1,170,219.72 | 38.14% | 1024 | 1,080,316.53 | 26.91% |
| block | 40 | 256 | 9,509.33 | 2.70% | 1024 | 9,273.03 | 1.02% |
| blocks | 256 | 256 | 293.84 | 2.71% | 512 | 289.15 | 1.54% |
| model | 256 | 256 | 294.21 | 2.76% | 512 | 289.69 | 1.58% |
| paged_attention | 2 | 4 | 1,279.07 | 88.26% | 64 | 1,028.46 | 91.43% |
| paged_model | 4 | 4 | 34.66 | 86.80% | 40 | 31.26 | 91.31% |

## Interpretation

1. **The full dense stack saturates around batch 256.** `blocks` and `model` peak at 293.84 and 294.21 tok/s, respectively, at batch 256. Batch 512 changes throughput by less than 2%. Batch 1024 exceeds the runtime memory limit for these two stages.

2. **The MLP needs the largest batch to saturate.** Its throughput scales to 1.17 million tok/s at batch 512, then falls by 7.7% at batch 1024. Its peak weighted DRAM utilization is only 38.14%, so peak throughput is not associated with saturating DRAM bandwidth.

3. **The shared-cache attention paths saturate early.** The isolated attention kernel peaks at batch 2. The full attention layer reaches 95% of its observed maximum by batch 4 and peaks at batch 40. Increasing batch size beyond that adds no throughput.

4. **Dense shared-cache DRAM utilization falls as batch grows.** At small batches, the attention and block paths show high instantaneous DRAM utilization. At large batches, reuse of the one physically shared K/V prefix amortizes DRAM traffic while compute and kernel execution grow. Consequently, the large-batch dense model peaks at only 2.76% duration-weighted DRAM utilization; it is not DRAM-bandwidth-bound at saturation in this shared-cache synthetic layout.

5. **The paged approximation is the clear memory-bandwidth bottleneck.** `paged_attention` and `paged_model` remain around 87-91% of peak DRAM throughput while their token throughput saturates by batches 2-4. Page gathering materializes dense K/V tensors, so it both consumes bandwidth and limits the largest feasible batch to 64 for paged attention and 40 for the paged model.

6. **The original batch-32 result hid two different regimes.** Batch 32 was already past saturation for isolated/shared attention and the paged stages, but it was too small to expose the batch-256 full-model plateau or the batch-512 MLP plateau.

## Artifacts

- Plot: `results/component_bw_10k_batch_sweep_report.png`
- Combined throughput CSV (remote): `results/component_bw_10k_batch_sweep_combined_throughput.csv`
- NCU stage summary (remote): `results/component_bw_10k_batch_sweep_report_ncu_summary.csv`
- NCU kernel-type summary (remote): `results/component_bw_10k_batch_sweep_report_ncu_kernel_types.csv`
- Generated report (remote): `results/component_bw_10k_batch_sweep_report.md`
- Remote run log: `results/component_bw_10k_batch_sweep_large.remote.log`

NCU uses kernel replay, so its profiled wall times are not throughput measurements. Throughput and DRAM-utilization values come from separate runs with the same stage, shape, and batch size.
