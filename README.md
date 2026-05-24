# GPU Memory Bandwidth Benchmarks

This project measures the memory-bandwidth behavior of LLM **decode-step** (single-token
generation) workloads on an NVIDIA GPU, using NVIDIA's profilers — Nsight Compute (`ncu`)
for per-kernel hardware counters and Nsight Systems (`nsys`) for whole-run GPU-metric
timelines.

There are two benchmark suites:

- **`attention_bw`** — isolated scaled-dot-product-attention (SDPA) kernels for a single
  decode step (one query token attending over a KV cache). Use this to compare attention
  backends in isolation.
- **`llm_bw`** — a full Hugging Face transformer model running the decode phase end to end.
  Use this to see where time and bandwidth go across the whole model (attention, linear
  layers, norms, etc.).

The local machine is assumed to have **no GPU**. Everything that needs CUDA runs on a remote
host over SSH; the `*_remote.sh` scripts sync the repo, run the profiler remotely, render the
plot remotely, and copy the resulting `.png` back. Locally you only need CPU/tooling
dependencies.

## Requirements

- **Local:** Python managed with [`uv`](https://docs.astral.sh/uv/), plus SSH/rsync access to
  the GPU host. Dependencies are in `requirements.txt` (numpy, pandas, matplotlib, transformers,
  accelerate, sentencepiece; `ruff`/`ty` for tooling). PyTorch is intentionally **not** installed
  locally.
- **Remote GPU host:** CUDA-capable GPU with a matching PyTorch build, `transformers`, and the
  NVIDIA profilers (`ncu` and/or `nsys`) on `PATH`. `flash_attention_2` / `flash_attention_3`
  require the corresponding packages installed on the remote.

Format, lint, and type-check with:

```bash
uvx ruff format .
uvx ruff check .
uvx ty check
```

## Remote configuration

The `*_remote.sh` scripts source `scripts/sync_remote.sh`, which takes the host and remote
directory as its first two positional arguments.

```bash
scripts/sync_remote.sh my-gpu-host ~/work
```

NOTE: Each profiling run automatically re-syncs the repo first.

## attention_bw — SDPA kernel benchmark

Compares attention backends for a single decode step. Shapes are `B,H,CACHE_SEQ,D`: the query
is a single token `(B,H,1,D)`, the K/V cache is `(B,H,CACHE_SEQ,D)`.

Kernels: `sdpa_math`, `sdpa_mem_efficient`, `sdpa_flash` (or `all`).

Run remotely with the profiler wrappers (output filename is timestamped automatically;
extra args after the host are forwarded to `main.py run`):

```bash
# Nsight Compute (per-kernel DRAM/SM counters)
scripts/run_ncu_remote.sh hinton-01 --kernels all --shape 2,64,4096,128 --dtype fp16

# Nsight Systems (GPU-metric timeline)
scripts/run_nsys_remote.sh hinton-01 --kernels sdpa_flash --shape 1,16,4096,64
```

Both scripts render the plot on the remote and copy a `.png` into `results/`.

## llm_bw — full-model decode benchmark

Runs a Hugging Face model: one prefill pass over the prompt, then token-by-token decode. The
attention implementation is selectable.

- Models (`--model`): `llama-7b`, `llama-13b`, `mistral-7b`, `phi-3-mini` (default `phi-3-mini`).
- Attention (`--attention`): `eager`, `sdpa` (default), `flash_attention_2`, `flash_attention_3`,
  `flex_attention`, and the `paged|*` variants.
- Key knobs: `--dtype`, `--prompt-length`, `--decode-tokens`, `--warmup-tokens`, `--batch-size`.

```bash
# Nsight Compute (per-layer-type breakdown)
scripts/run_llm_ncu_remote.sh hinton-01 --model phi-3-mini --attention sdpa --prompt-length 512

# Nsight Systems (timeline over the decode phase)
scripts/run_llm_nsys_remote.sh hinton-01 --model mistral-7b --attention flash_attention_2
```

> NCU runs every kernel multiple times to collect counters, so it is slow. The LLM NCU wrapper
> caps decode to 1 token; keep token counts small when profiling with `ncu`.

## Profiling details

- **NVTX ranges** wrap warmup and measured iterations (`...:warmup`, `...:iter`, `...:case`). The
  profiler wrappers filter on these so that warmup work is excluded from the reported metrics.
- **Nsight Compute** collects: `dram__bytes_read.sum`, `dram__bytes_write.sum`,
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__time_duration.sum`. The visualizer
  pivots these per kernel and classifies kernels into types (attention / linear / norm / … for
  `llm_bw`; the three SDPA backends for `attention_bw`).
- **Nsight Systems** captures GPU-metric timelines (DRAM throughput %, SMs-active %) plus the CUDA
  kernel trace, exported to SQLite for plotting bandwidth and SM utilization over time.

## Visualization

The remote wrappers visualize automatically. To re-render from a raw profiler artifact (CSV from
`ncu`, `.sqlite` from `nsys`) directly:

```bash
# attention_bw
uv run main.py visualize results/attention_bw_ncu_<ts>.csv  -o out.png
uv run main.py visualize results/attention_bw_nsys_<ts>.sqlite -o out.png

# llm_bw (config flags only annotate the plot title)
uv run llm_main.py visualize results/llm_bw_ncu_<ts>.csv -o out.png \
  --model phi-3-mini --dtype fp16 --attention sdpa --prompt-length 512
```

The file extension selects the path: `.csv` → Nsight Compute view, `.sqlite` → Nsight Systems
timeline view.

## Running directly on a GPU host

The `*_remote.sh` scripts are thin wrappers around `scripts/run_{ncu,nsys}.sh` and
`scripts/run_llm_{ncu,nsys}.sh`. On a machine that already has a GPU you can invoke those (or
`main.py` / `llm_main.py`) directly:

```bash
uv run main.py run --kernels all --shape 2,64,4096,128
uv run llm_main.py run --model phi-3-mini --decode-tokens 50
```
