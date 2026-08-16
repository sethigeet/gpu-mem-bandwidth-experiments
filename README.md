# GPU Memory Benchmarks for LLM Inference

This repository measures how LLM inference workloads use NVIDIA GPU memory bandwidth. It covers
isolated attention kernels, full-model decode, a staged synthetic model, prefix-cache locality,
and vLLM serving under request load.

The benchmark code is organized under one package and exposed through one command:

| Suite | Purpose |
| --- | --- |
| `attention` | Compare scaled-dot-product attention backends for one decode step. |
| `model` | Profile token-by-token decode in a Hugging Face causal language model. |
| `components` | Isolate the cost of attention, MLPs, decoder blocks, full models, and paged KV access. |
| `prefix-cache` | Reproduce prefix-homogeneity and locality experiments with vLLM. |
| `vllm` | Measure serving throughput, GPU utilization, scheduling policies, and EngineCore overhead. |

Checked-in Markdown files under [`docs/`](docs/) are curated studies. Analysis code writes raw
CSV tables and figures; it does not generate or overwrite narrative reports.

## Environment

Python environments are managed with [uv](https://docs.astral.sh/uv/). The local development
machine does not need CUDA or PyTorch:

```bash
uv sync --group dev
uv run gpu-memory-benchmarks --help
uvx ruff format --check .
uvx ruff check .
uvx ty check
```

On a CUDA GPU host, install the model dependencies needed by the attention, model, and component
suites:

```bash
uv sync --extra model
```

The serving scripts can install the pinned vLLM version remotely, or it can be installed with:

```bash
uv sync --extra model --extra serving
```

Nsight Compute (`ncu`) and Nsight Systems (`nsys`) must be available on the GPU host. Some
attention implementations also require their corresponding optional packages.

## Remote execution

Every GPU-facing wrapper accepts `--remote`, `--host`, and `--remote-dir`. The wrapper synchronizes
the project, runs the benchmark and visualization remotely, then copies the requested artifacts
back. Set defaults once if preferred:

```bash
export REMOTE_HOST=gpu-host
export REMOTE_DIR=~/gpu-memory-benchmarks
scripts/remote.sh sync
```

Long component and vLLM jobs support `--detach`; they run in remote tmux and print the session,
log, output prefix, and fetch command.

## Attention and full-model decode

The shared profiling wrapper removes duplicated Nsight setup between the two suites:

```bash
# Isolated SDPA kernels under Nsight Compute.
scripts/profile.sh attention ncu --remote -- \
  --kernels all --shape 2,64,4096,128 --dtype fp16

# One model's decode phase under Nsight Systems.
scripts/profile.sh model nsys --remote -- \
  --model mistral-7b --attention flash_attention_2 \
  --prompt-length 512 --batch-size 1
```

Attention shapes use `B,H,CACHE_SEQ,D`. Full-model aliases include `llama-7b`, `llama-13b`,
`llama-3-8b`, `llama-3.1-8b`, `mistral-7b`, and `phi-3-mini`; a full Hugging Face model ID is
also accepted. NCU profiles only one measured model token because counter replay is expensive.

Measured NVTX ranges use `gpu_memory:attention:...` and `gpu_memory:model:...`.

## Synthetic decode components

The component ladder separates increasingly complete decode paths:

- `attention_kernel`: direct SDPA over a preallocated KV cache.
- `attention_layer`: QKV/output projections plus SDPA.
- `mlp`: gated feed-forward network only.
- `block`: one normalized attention/MLP decoder block with residuals.
- `decoder_stack`: all decoder blocks.
- `full_model`: embeddings, decoder stack, final norm, LM head, and sampling.
- `paged_attention`: attention with a PyTorch block-table KV gather.
- `paged_full_model`: the full synthetic model with the same paged-KV approximation.

KV layouts are `replicated`, `shared`, and `paged`. Replicated prefixes are unique per request;
shared and paged prefixes refer to one physical prefix. The explicitly paged stages honor the
requested prefix-sharing mode while always using block-table storage. Start with a smoke run:

```bash
scripts/components.sh nsys --remote -- \
  --smoke --stages attention_kernel attention_layer mlp block paged_attention
```

Run a large throughput and NCU batch sweep in detached tmux:

```bash
scripts/components.sh sweep-bundle --remote --detach \
  --out results/components_10k_batch_sweep -- \
  --model phi-3-mini --prefix-len 10000 --decode-tokens 64 --layout shared \
  --batch-sizes 1 2 4 8 16 32 40 48 64 128 256 512 1024
```

For short, non-shared prompts, use the replicated layout. Larger batches are useful because the
shorter attention path can keep scaling after the long-prefix sweep has saturated:

```bash
scripts/components.sh sweep-bundle --remote --detach \
  --out results/components_128_unique_batch_sweep -- \
  --model phi-3-mini --prefix-len 128 --decode-tokens 64 --layout replicated \
  --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
```

The NCU sweep uses kernel replay by default and checkpoints every stage/batch pair with `.done`
or `.skipped`. Infeasible paged points do not abort the dense-stage sweep. Override the NCU batch
list with `COMPONENT_BATCH_SIZES` when needed. Set `COMPONENT_NCU_PROFILE=memory-hierarchy` for a
diagnostic counter pass that adds L1/L2 hit rates, HBM sectors, occupancy, SM compute/pipe
throughput, and memory- versus math-stall indicators.

Combine a throughput CSV and per-stage NCU files into data and a figure:

```bash
uv run gpu-memory-benchmarks components analyze \
  --throughput-csv results/components_10k_batch_sweep_throughput.csv \
  --ncu-glob 'results/components_10k_batch_sweep_ncu_b*_*.csv' \
  --output-prefix results/components_10k_analysis
```

This writes `components_10k_analysis_stage_summary.csv`,
`components_10k_analysis_kernel_types.csv`, and `components_10k_analysis.png`.

## Prefix-cache locality

These experiments construct raw token sequences whose leading tokens are physically shared by
vLLM's prefix cache. Each command writes a throughput CSV and plot:

```bash
scripts/prefix-cache.sh sweep homogeneity --remote -- \
  --model llama-7b --num-requests 256

scripts/prefix-cache.sh sweep prefix-length --remote -- --total-len 4096
scripts/prefix-cache.sh sweep num-groups --remote -- --values 1,2,4,8,16,32
scripts/prefix-cache.sh sweep batch-size --remote -- \
  --values 16,32,64,128,256 --hetero-groups 5
```

Use the `nsys` command with one sweep value to inspect bandwidth directly:

```bash
scripts/prefix-cache.sh nsys homogeneity --remote -- --values 1.0
scripts/prefix-cache.sh nsys homogeneity --remote -- --values 0.5
```

Measured ranges use `gpu_memory:prefix_cache:...`.

## vLLM serving and scheduling

Profile a server while `vllm bench serve` supplies request load:

```bash
scripts/vllm-server.sh profile serve --remote --detach -- \
  --model phi-3-mini \
  --random-input-len 2048 --random-output-len 64 \
  --num-prompts 256 --max-num-seqs 256 --request-rate inf
```

The measured serving window is `gpu_memory:vllm:serve:bench`, excluding startup and warmup. Fetch
the output prefix printed by the detached launcher:

```bash
scripts/vllm-server.sh fetch --out results/vllm_serve_nsys_<timestamp>
```

Compare paired async/sync scheduling trials:

```bash
scripts/vllm-server.sh compare --remote --detach -- \
  --model phi-3-mini --random-input-len 2048 --random-output-len 64 \
  --num-prompts 256 --max-num-seqs 256 --max-concurrency 256 \
  --request-rate inf --repetitions 5
```

The comparison writes trial, summary, paired-comparison, policy-comparison, and impact CSVs plus a
comparison figure. Custom `radix_cost`, `chunked_hash_tree_python`,
`chunked_hash_tree_cpp`, and `chunked_hash_tree_bandit` policies live under
`gpu_memory_benchmarks/serving/schedulers/`. Install the pinned policy-enabled vLLM environment
with:

```bash
scripts/vllm-server.sh install-policies --remote --detach
```

The default isolated environment is `~/.cache/gpu-memory-benchmarks/vllm/policy_venv`.
`VLLM_BENCH_PROFILE_SCHEDULER=1` enables scheduler timing and
`VLLM_BENCH_PROFILE_INTERVAL` controls cumulative logging.

### Detailed EngineCore timing

Instrumentation targets upstream vLLM 0.22.1. Install it on the GPU host, then profile CPU phase
timings or CUDA model execution:

```bash
scripts/vllm-server.sh install-timing --remote

scripts/vllm-server.sh profile overhead --remote --detach \
  --out results/vllm_step_breakdown_sync -- \
  --model llama-3.1-8b --scheduling-mode sync \
  --max-model-len 10240 --random-prefix-len 10000 --random-input-len 20 \
  --random-output-len 50 --num-prompts 300 --max-num-seqs 100 \
  --max-concurrency 100 --request-rate inf

scripts/vllm-server.sh profile model-gpu --remote --detach \
  --out results/vllm_model_gpu_sync -- \
  --model llama-3.1-8b --scheduling-mode sync \
  --max-model-len 10240 --random-prefix-len 10000 --random-input-len 20 \
  --random-output-len 50 --num-prompts 300 --max-num-seqs 100 \
  --max-concurrency 100 --request-rate inf
```

CPU timing, py-spy, and NSYS runs should remain separate because combining them changes the
measurement. CPU `execute_model` duration measures asynchronous host submission; CUDA events are
used when actual GPU duration is required.

## Direct CLI and visualization

On a configured GPU host, the wrappers are optional:

```bash
uv run gpu-memory-benchmarks attention run --kernels all --shape 2,64,4096,128
uv run gpu-memory-benchmarks model run --model phi-3-mini --decode-tokens 50
uv run gpu-memory-benchmarks prefix-cache homogeneity \
  -o results/prefix_homogeneity.csv --model llama-7b
```

Existing profiler artifacts can be rendered without a GPU:

```bash
uv run gpu-memory-benchmarks attention visualize results/attention_ncu_<timestamp>.csv -o out.png
uv run gpu-memory-benchmarks model visualize results/model_nsys_<timestamp>.sqlite -o out.png
uv run gpu-memory-benchmarks vllm summarize trace.sqlite -o summary.csv
```

## Studies

- [Synthetic component saturation](docs/component_saturation_study.md)
- [vLLM scheduling and request policies](docs/vllm_scheduling_study.md)
- [vLLM attention backend comparison](docs/vllm_attention_backend_study.md)
- [vLLM CPU overhead investigation](docs/vllm_cpu_overhead_study.md)
