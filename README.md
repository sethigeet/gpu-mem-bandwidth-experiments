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
- **`prefix_bw`** — reproduces the prefix-homogeneity claims of the *Feather* paper
  ("Requests of a Feather Must Flock Together", `Cache_aware_LLM_batching.pdf`) on vLLM with
  prefix caching. Builds batches of requests that physically share KV-cache prefixes and
  measures how decode throughput and DRAM bandwidth change with homogeneity, shared-prefix
  length, number of prefix groups, and batch size.
- **`vllm_bw`** — profiles a standard `vllm serve` OpenAI-compatible server while
  `vllm bench serve` sends many requests. Use this to answer whether a realistic serving run is
  saturating DRAM during decode-heavy request load or leaving memory-bandwidth headroom.

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

## component_bw — staged synthetic decode ladder

Builds the missing pieces between isolated attention and vLLM with PyTorch modules shaped like
Phi-3-mini by default. Each stage runs the same long shared-prefix decode shape and reports
decode throughput plus estimated KV bytes and launch count. Nsight ranges use
`component_bw:<stage>:case|warmup|iter`.

Stages:

- `attention_kernel` — direct SDPA decode over preallocated KV.
- `attention_layer` — QKV/O projections plus SDPA.
- `mlp` — gated MLP only.
- `block` — RMSNorm, attention, residuals, and MLP for one decoder block.
- `blocks` — repeated decoder blocks.
- `model` — embeddings, all blocks, final norm, lm head, and argmax sampling.
- `paged_attention` — attention layer with a PyTorch block-table KV gather.
- `paged_model` — full synthetic model with the same paged-KV approximation.

Run a small remote smoke test first:

```bash
scripts/run_component_nsys_remote.sh hinton-01 ~/code/attention-bw \
  --smoke \
  --stages attention_kernel attention_layer mlp block paged_attention
```

Run the canonical 10K shared-prefix matrix:

```bash
scripts/run_component_nsys_remote_tmux.sh hinton-01 ~/code/attention-bw \
  --model phi-3-mini \
  --prefix-len 10000 \
  --decode-tokens 64 \
  --batch-size auto \
  --max-auto-batch 256 \
  --layout shared
```

The tmux launcher prints a `scripts/fetch_component_nsys_remote.sh ...` command to copy artifacts
back after the detached session finishes.

Profile one stage with Nsight Compute counters:

```bash
scripts/run_component_ncu_remote.sh hinton-01 ~/code/attention-bw \
  --stage attention_kernel \
  --prefix-len 10000 \
  --batch-size auto \
  --layout shared
```

The NSYS wrapper writes `results/component_bw_nsys_<ts>.csv` for throughput,
`_summary.png` for the stage waterfall, `.png` for DRAM/SM timelines, and
`_nsys_summary.csv` with average/p50/p95/max utilization inside the measured component ranges.

## prefix_bw — Feather prefix-homogeneity reproduction

Reproduces the paper's claim that, because decode is memory-bandwidth bound, batches whose
requests **physically share a KV-cache prefix** get better spatial/temporal locality (higher
effective DRAM bandwidth, fewer bytes fetched) and so higher decode throughput. Requests are
built as raw token-id lists; identical leading tokens make vLLM's prefix cache store the shared
prefix once and let every request in the group read the same KV blocks.

vLLM is a GPU-host-only dependency. Install it once into the remote venv:

```bash
scripts/install_vllm_remote.sh        # uv pip install vllm on the remote
```

Each subcommand sweeps one knob and writes a throughput CSV + plot. The `EXPERIMENT` is one of
`homogeneity` (Fig 4), `prefix-length` (Fig 5), `num-groups` (Fig 6), `batch-size` (Figs 8–9):

```bash
# Fig 4: vary the fraction of requests on a shared prefix (homogeneous beta=0/1 vs mixed)
scripts/run_prefix_remote.sh homogeneity --model llama-7b --num-requests 256

# Fig 5: vary the shared prefix length
scripts/run_prefix_remote.sh prefix-length --total-len 4096

# Fig 6: vary the number of distinct prefix groups
scripts/run_prefix_remote.sh num-groups --values 1,2,4,8,16,32

# Figs 8-9: sweep batch size for homogeneous vs heterogeneous workloads (two lines)
scripts/run_prefix_remote.sh batch-size --values 16,32,64,128,256 --hetero-groups 5
```

To verify the **bandwidth** claim directly (not just throughput), profile a single config under
nsys and render the DRAM-bandwidth timeline (reusing the `llm_bw` nsys visualizer). Pass one
sweep value so the timeline is clean — e.g. fully homogeneous vs mixed:

```bash
scripts/run_prefix_nsys_remote.sh homogeneity --values 1.0   # homogeneous: high BW
scripts/run_prefix_nsys_remote.sh homogeneity --values 0.5   # mixed: lower BW
```

Common knobs (all subcommands): `--model` (registry key or raw HF id), `--dtype`,
`--num-requests`, `--decode-tokens`, `--max-num-seqs` (vLLM batch size),
`--gpu-memory-utilization`, `--values` (the sweep points), `--no-warmup`. The measured
`generate` is wrapped in `prefix_bw:...:case` NVTX ranges (with `...:warmup` around prefix-cache
warmup) so the nsys visualizer isolates decode, exactly like `llm_bw`. Defaults are scaled down
from the paper (which used Llama-3-8B with 10K-token prefixes); raise `--prefix-len` /
`--total-len` / `--num-requests` toward those to match it more closely.

## vllm_bw — vLLM serving under request load

Runs `vllm serve` under Nsight Systems, waits for `/health`, warms the server with a small
`vllm bench serve` run, then wraps the measured benchmark in an NVTX range named
`vllm_bw:serve:bench`. The visualizer filters the exported SQLite to that range so startup,
model load, and warmup do not dilute the DRAM-bandwidth numbers.

```bash
# Start a standard serving profile on the remote GPU host in detached tmux.
scripts/run_vllm_serve_nsys_remote.sh hinton-01 ~/code/attention-bw \
  --model phi-3-mini \
  --random-input-len 2048 \
  --random-output-len 64 \
  --num-prompts 256 \
  --max-num-seqs 256 \
  --request-rate inf

# The start command prints the output prefix. Fetch artifacts after tmux finishes.
scripts/fetch_vllm_serve_nsys_remote.sh hinton-01 ~/code/attention-bw \
  results/vllm_bw_serve_nsys_<ts>
```

The wrapper writes:

- `results/vllm_bw_serve_nsys_<ts>.png` — DRAM and SM utilization timelines for the measured
  request-load window.
- `results/vllm_bw_serve_nsys_<ts>_summary.csv` — per-metric average, p50, p95, max, and
  `headroom_vs_p95_pct`.
- `results/vllm_bw_serve_nsys_<ts>_logs/bench.log` — `vllm bench serve` throughput, TTFT, TPOT,
  and inter-token-latency output.

Interpretation: if DRAM p95 is close to sustained peak while SM activity is materially lower,
decode is behaving as memory-bound and remaining gains likely need better memory locality,
batching, KV-cache layout, or quantization. If DRAM p95 is far below peak during the measured
window, there is likely serving/runtime overhead, insufficient concurrency, model-size effects,
or another bottleneck leaving bandwidth on the table. Increase `--num-prompts`,
`--max-num-seqs`, or `--max-concurrency` to push harder before concluding the kernel path itself
is not bandwidth-limited.

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
uv run prefix_main.py homogeneity -o results/prefix_homo.csv --model llama-7b
```
