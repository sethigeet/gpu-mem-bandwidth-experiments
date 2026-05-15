# Decode Attention Kernel Memory Bandwidth

This project measures memory-bandwidth behavior for decode-stage attention kernels on an NVIDIA GPU host reachable over SSH. Local dependencies stay CPU/tooling-only; CUDA and attention-kernel packages are installed only on the remote machine through `uv`.

## Remote Setup

Default host is `hinton-01` and default remote directory is `~/attention-bw`.

```bash
./scripts/bootstrap_remote.sh
```

The bootstrap installs project dependencies from `pyproject.toml`, then installs PyTorch on the remote only. For a CUDA 12.1 PyTorch wheel instead of CUDA 12.4:

```bash
CUDA_REQ=cu121 ./scripts/bootstrap_remote.sh hinton-01
```

Override the remote Torch package constraint with `TORCH_REQ`:

```bash
TORCH_REQ="torch==2.6.0" CUDA_REQ=cu124 ./scripts/bootstrap_remote.sh hinton-01
```

Optional kernels can be installed on the remote by passing pip packages through `INSTALL_OPTIONAL`:

```bash
INSTALL_OPTIONAL="flash-attn --no-build-isolation xformers" ./scripts/bootstrap_remote.sh hinton-01
```

`flash-attn` often needs a matching CUDA toolkit/compiler on the remote. If installation fails, use PyTorch SDPA kernels first.

## Run Benchmarks

```bash
./scripts/run_remote.sh hinton-01
```

Run specific shapes and kernels:

```bash
./scripts/run_remote.sh hinton-01 \
  --kernels sdpa_math sdpa_mem_efficient sdpa_flash \
  --shape 1,16,2048,64 \
  --shape 1,16,4096,64 \
  --dtype fp16 \
  --iters 100 \
  --out results/hinton_attention_bw.csv
```

Shapes are specified as `B,H,CACHE_SEQ,D`. The benchmark creates a single-token query tensor with shape `B,H,1,D` and KV-cache tensors with shape `B,H,CACHE_SEQ,D`, matching the attention work in one decode step.

If optional packages are installed:

```bash
./scripts/run_remote.sh hinton-01 --kernels sdpa_flash flash_attn xformers --shape 1,16,8192,64
```

## Nsight Compute DRAM Metrics

The Python benchmark estimates effective bandwidth from tensor traffic and latency. For hardware DRAM counters, run Nsight Compute on the remote if `ncu` is installed or available through a module:

```bash
./scripts/run_ncu_remote.sh hinton-01 --kernels sdpa_flash --shape 1,16,4096,64
```

This writes `results/attention_bw_ncu.csv` on the remote host.

## Interpreting Results

The table includes:

- `median_ms`: CUDA-event median kernel time.
- `effective_gb_s`: estimated bytes moved divided by median time.
- `utilization_pct_of_peak`: estimated bandwidth divided by peak bandwidth inferred from `nvidia-smi`.
- `tflops`: rough attention matmul FLOP rate, excluding softmax overhead.

For fused kernels such as FlashAttention, `effective_gb_s` is based on single-token Q/O traffic plus KV-cache traffic, not full score-matrix materialization. For `sdpa_math`, the estimate includes read/write traffic for the materialized attention score/probability matrices. Treat these as comparable effective-bandwidth indicators, not exact HBM byte counters. Use `run_ncu_remote.sh` when exact DRAM counter data is required.

If `nvidia-smi` cannot report memory clock and bus width, set peak bandwidth manually:

```bash
ssh hinton-01 'cd ~/attention-bw && ATTN_BW_PEAK_GB_S=1555 uv run attention-bw --shape 1,16,4096,64'
```
