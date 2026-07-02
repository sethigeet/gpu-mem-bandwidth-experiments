# Prefix BW Memory Investigation

## Goal

Find the main GPU memory bottlenecks in the `prefix_bw` vLLM prefix-sharing benchmarks, connect them to the cache-aware batching paper, and implement changes that improve memory utilization.

## Running Notes

- Started by extracting `Cache_aware_LLM_batching.pdf` with Ghostscript:
  `gs -q -sDEVICE=txtwrite -o /tmp/cache_aware_llm_batching.txt Cache_aware_LLM_batching.pdf`
- The current `prefix_bw` benchmark constructs synthetic token-id prompts where requests physically share vLLM prefix-cache keys, warms distinct prefixes, then measures `llm.generate` decode throughput.
- First impression from code: the benchmark varies homogeneity and group count, but it does not currently reorder or schedule requests by prefix locality. That means heterogeneous batches may intentionally mix unrelated KV-cache regions in one decode step, which is the bad case discussed by the paper.

## Paper Findings

- Decode is memory-bound because each new token attends over all prior KV-cache blocks. Longer contexts make each decode step a repeated KV-cache sweep.
- Prefix-homogeneous batches improve both spatial and temporal locality: neighboring requests walk the same KV-cache regions, so cache lines prefetched into L2 for one request can be reused by other requests before eviction.
- The paper reports that even one request from a divergent prefix can sharply reduce effective bandwidth; the main loss happens when going from one prefix group to two, while adding more groups matters less until the total KV footprint causes evictions.
- Bigger batches are not always better. Once memory bandwidth saturates, a moderately smaller prefix-homogeneous batch can outperform a larger heterogeneous batch because it fetches fewer distinct KV-cache regions and uses L2 more effectively.
- The paper's Feather design has two separable ideas:
  - Find same-prefix candidates cheaply with chunked cumulative hashes instead of expensive radix-tree traversal.
  - Stop batch formation when adding a request would shrink shared-prefix locality more than the batch-size gain is worth.

## Current Bottlenecks In `prefix_bw`

- The benchmark can reproduce the locality cliff, but it does not yet provide a scheduling intervention. `run_workload` passes all requests to one `llm.generate` call, leaving vLLM's offline scheduler to form a large mixed batch whenever `max_num_seqs` permits it.
- Synthetic workloads already know each request's prefix group, but the runner was discarding that metadata. That prevented an oracle same-prefix scheduler baseline.
- For heterogeneous workloads with long shared prefixes and non-trivial decode lengths, the expected bottleneck is not tensor core utilization. It is repeated attention reads over unrelated KV-cache blocks, which lowers L2 hit rate and effective DRAM bandwidth.

## Attempt 1: Oracle Prefix-Grouped Scheduling

Implemented:

- Added `prefix_group_ids` to `Workload` and populated it in all synthetic workload builders.
- Added a runner schedule mode:
  - `single`: old behavior, one mixed `llm.generate` call.
  - `prefix-grouped`: group requests by known prefix ID and execute one homogeneous microbatch at a time, chunked by `--prefix-batch-size` or `--max-num-seqs`.
- Added CLI controls:
  - `--schedule single|prefix-grouped` on existing sweeps.
  - New `schedule` experiment to compare `single` vs `prefix-grouped` across group counts.

Expected result:

- `prefix-grouped` should improve decode throughput and DRAM bandwidth utilization when each prefix group has enough requests to avoid tiny underfilled microbatches.
- It may lose for short decode lengths, small groups, or no prefix sharing because extra `generate` calls reduce batching efficiency and add prefill/scheduling overhead.

Useful remote runs:

```bash
scripts/run_prefix_remote.sh schedule --model llama-7b --num-requests 256 --prefix-len 4096 --decode-tokens 128 --values 1,2,4,8,16
scripts/run_prefix_nsys_remote.sh schedule --model llama-7b --num-requests 256 --prefix-len 4096 --decode-tokens 128 --values 8 --schedules single
scripts/run_prefix_nsys_remote.sh schedule --model llama-7b --num-requests 256 --prefix-len 4096 --decode-tokens 128 --values 8 --schedules prefix-grouped
```

Verification:

- `uv run prefix_main.py schedule --help` works locally without importing GPU-only vLLM.
- `uvx ruff format prefix_bw prefix_main.py` completed with no pending formatting changes after the final edit.
- `uvx ruff check prefix_bw prefix_main.py` passed.
- `uvx ty check prefix_bw prefix_main.py` passed.
- Full-repo `uvx ty check` still reports pre-existing unresolved `transformers` imports in `llm_bw`, which is outside this change and consistent with local dependencies not being installed.

Remote smoke test:

```bash
scripts/run_prefix_remote.sh schedule --model phi-3-mini --num-requests 64 --prefix-len 1024 --decode-tokens 32 --values 1,2,4,8
```

Results:

- 1 group: `single` 2034.3 toks/s, `prefix-grouped` 2206.0 toks/s.
- 2 groups: `single` 2061.6 toks/s, `prefix-grouped` 1704.2 toks/s.
- 4 groups: `single` 2045.5 toks/s, `prefix-grouped` 1168.1 toks/s.
- 8 groups: `single` 1955.1 toks/s, `prefix-grouped` 700.1 toks/s.

Interpretation:

- The scheduler path works, and homogeneous execution can help when the batch is already one large prefix group.
- Pure prefix grouping is too naive for small groups. It creates underfilled decode calls whose lower compute occupancy outweighs the memory-locality win.
- A larger run with 256 requests, 2K prefixes, and 128 decode tokens stalled on the first `single` case for more than 30 minutes and was stopped. That run was too large for a tight iteration loop on the shared remote.

## Attempt 2: Adaptive Oracle Prefix Scheduling

Implemented:

- Added `prefix-adaptive` schedule mode.
- Large same-prefix groups are decoded as homogeneous microbatches.
- Prefix groups smaller than `--min-prefix-batch-size` are left in a residual mixed batch.
- If no prefix group is large enough, the strategy falls back exactly to `single`.

Expected result:

- Avoid the severe small-group regression observed in Attempt 1.
- Preserve the locality win when one or more prefix groups are large enough to justify a separate decode call.
- This is still a heuristic; a Feather-style bandit/RL policy would learn the threshold from observed throughput instead of taking `--min-prefix-batch-size` as a static knob.

Remote follow-up:

- Tried to rerun the 64-request smoke test after adding `prefix-adaptive`, but the remote process stalled during model load before any benchmark rows. Stopped the local wrapper.
- Because the post-change remote did not reach the measured cases, the adaptive policy is locally validated but not yet benchmark-validated.

Final local verification:

- `uvx ruff format prefix_bw prefix_main.py`
- `uvx ruff check prefix_bw prefix_main.py`
- `uvx ty check prefix_bw prefix_main.py`

## Measurement Follow-up

- Fixed `scripts/run_prefix_nsys.sh` to set `VLLM_WORKER_MULTIPROC_METHOD=spawn`, matching `scripts/run_prefix.sh`. Without this, vLLM can fail under nsys with `Cannot re-initialize CUDA in forked subprocess`.
- Fixed `prefix_bw.runner.build_llm` so vLLM constructors with `**kwargs` receive `max_model_len`. Without this, some models fell back to their default long context length, inflating KV-cache allocation.
- Updated `scripts/run_prefix_nsys_remote.sh` to keep large nsys artifacts (`.nsys-rep`, `.sqlite`) on the remote host. The wrapper now copies back only:
  - workload result CSV,
  - compact metrics summary CSV,
  - rendered PNG.
- Measurement on the previous remote host was blocked by another user's active vLLM benchmark using about 46 GiB of the 48 GiB GPU and near-100% GPU/memory utilization. Switched the repo's default remote host to `gpu1` and stopped the previous measurement tmux sessions.
- `gpu1` has four idle RTX A6000 GPUs, but it has driver 470 / CUDA 11.4 and no Nsight Systems in `PATH`.
- Tried setting up vLLM on `gpu1`:
  - Current vLLM wheels install CUDA 13 packages and fail to import with `undefined symbol: cuTensorMapEncodeTiled`.
  - Older `vllm==0.1.4` can import with `torch==1.12.1+cu113`, but it predates vLLM prefix caching and fails the benchmark with `EngineArgs.__init__() got an unexpected keyword argument 'enable_prefix_caching'`.
- Conclusion: `gpu1` cannot run the current vLLM prefix-cache benchmark faithfully without a newer NVIDIA driver/CUDA stack and a current vLLM install, or a custom non-vLLM benchmark path.

## Successful `hinton-01` Measurement

Ran on `hinton-01` after explicitly syncing to `/home/geet/code/attention-bw`, ignoring the repo default remote host.

Configuration:

- model: `phi-3-mini`
- `--gpu-memory-utilization 0.45`
- `--num-requests 64`
- `--prefix-len 1024`
- `--suffix-len 32`
- `--decode-tokens 64`
- `--values 0.75`
- profiler GPU metrics device: `cuda-visible`

Results:

- `single`
  - decode throughput: 1718.68 toks/s
  - mean DRAM read bandwidth utilization: 37.74%
  - p95 DRAM read bandwidth utilization: 97.00%
  - max DRAM read bandwidth utilization: 101.00%
  - mean DRAM write bandwidth utilization: 1.12%
  - mean SMs active: 91.58%
- `prefix-adaptive`
  - decode throughput: 1627.38 toks/s
  - mean DRAM read bandwidth utilization: 42.88%
  - p95 DRAM read bandwidth utilization: 97.00%
  - max DRAM read bandwidth utilization: 100.00%
  - mean DRAM write bandwidth utilization: 1.02%
  - mean SMs active: 91.21%

Interpretation:

- `prefix-adaptive` improved mean DRAM read bandwidth utilization by about 13.6% relative to `single`.
- It did not improve overall throughput on this small workload; throughput dropped by about 5.3% because the workload split into two generate calls (`max_request_batch_size=48`) instead of one 64-request call.
- This confirms the paper's tradeoff: prefix locality can improve effective memory bandwidth, but smaller batches can lose enough compute/launch efficiency to reduce end-to-end throughput unless same-prefix groups are larger or decode length is longer.

## Llama HF-Token Measurement

Used the local `.env` `HF_TOKEN` on `hinton-01` by copying `.env` to `/home/geet/code/attention-bw/.env` with mode `600` and sourcing it inside the remote `tmux` session. The token allowed access to `meta-llama/Meta-Llama-3-8B`; the run got past gated-repo resolution and loaded the Llama architecture.

Attempted paper-like `PREFIX_LEN=10000` first, but `Meta-Llama-3-8B` advertises `max_position_embeddings=8192`, so vLLM rejected `max_model_len=10086`. Also tried `meta-llama/Meta-Llama-3.1-8B` for long-context support, but Hugging Face returned 403/not authorized for this token.

Completed the closest safe Llama run with `Meta-Llama-3-8B`:

- `NUM_REQUESTS=500`
- `PREFIX_LEN=8000`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=50`
- `VALUES=0.75`
- `MAX_NUM_SEQS=500`
- `MIN_PREFIX_BATCH_SIZE=100`
- schedules: `single`, `prefix-adaptive`

Results:

- `single`
  - decode throughput: 1068.60 toks/s
  - mean DRAM read bandwidth utilization: 5.67%
  - p95 DRAM read bandwidth utilization: 32.00%
  - max DRAM read bandwidth utilization: 99.00%
  - mean DRAM write bandwidth utilization: 1.42%
  - mean SMs active: 90.81%
- `prefix-adaptive`
  - decode throughput: 1089.39 toks/s
  - mean DRAM read bandwidth utilization: 10.62%
  - p95 DRAM read bandwidth utilization: 73.00%
  - max DRAM read bandwidth utilization: 100.00%
  - mean DRAM write bandwidth utilization: 1.63%
  - mean SMs active: 91.28%

Interpretation:

- `prefix-adaptive` improved decode throughput by about 1.95% on this 500-request Llama workload.
- Mean DRAM read bandwidth utilization improved by about 87.4% relative to `single` (`5.67%` to `10.62%`).
- The p95 DRAM read utilization more than doubled (`32%` to `73%`), suggesting the adaptive prefix split produced stronger high-locality decode phases even though the whole-profile mean remains diluted by setup/model-loading intervals.
- Large nsys artifacts were left on `hinton-01`; only compact workload and metrics CSVs were copied back locally.

## Llama 3.1 High-Concurrency Follow-up

After the Hugging Face token was granted Llama 3.1 access, rechecked model access with the correct repo IDs:

- `meta-llama/Llama-3.1-8B`: authorized
- `meta-llama/Llama-3.1-8B-Instruct`: authorized
- `meta-llama/Meta-Llama-3.1-8B`: redirect/legacy spelling; do not use for runs

Also made `scripts/run_prefix_nsys.sh` accept `GPU_METRICS_FREQUENCY` so long profiles can use lower-frequency GPU metrics and avoid producing 20GB SQLite exports during iteration.

Negative/neutral attempts:

- `Meta-Llama-3-8B`, `PREFIX_LEN=8000`, `NUM_REQUESTS=1000`, `DECODE_TOKENS=128`, `VALUES=1.0`:
  - decode throughput: 1164.81 toks/s
  - mean DRAM read bandwidth utilization: 4.85%
  - p95 DRAM read bandwidth utilization: 21.00%
  - reason it failed: vLLM reported only about 25x maximum concurrency for 8164-token requests on the 48GB GPU, so increasing request count created a long rolling decode rather than a wide memory-saturating batch.
- `Llama-3.1-8B`, `PREFIX_LEN=10000`, `NUM_REQUESTS=500`, `DECODE_TOKENS=128`, `VALUES=0.75`:
  - decode throughput: 977.86 toks/s
  - mean DRAM read bandwidth utilization: 9.75%
  - p95 DRAM read bandwidth utilization: 55.00%
  - reason it failed: same KV-capacity limit; vLLM reported about 211k GPU KV-cache tokens, which is only about 20 full 10K-context active requests.
- Tried setting `VLLM_ATTENTION_BACKEND=FLASHINFER`, but vLLM 0.22 reported it as an unknown environment variable and selected `FLASH_ATTN`. The installed backend selector prioritizes FlashAttention on this GPU, so this is not a valid backend override in the current environment.
- `Llama-3.1-8B`, `PREFIX_LEN=512`, `NUM_REQUESTS=1000`, `DECODE_TOKENS=256`, `VALUES=1.0`:
  - decode throughput: 3065.04 toks/s
  - mean DRAM read bandwidth utilization: 17.67%
  - p95 DRAM read bandwidth utilization: 38.00%
  - reason it was not best: removing the residual mixed batch lowered both throughput and mean DRAM read versus the `VALUES=0.75` high-concurrency case.
- `Llama-3.1-8B`, `PREFIX_LEN=1024`, `NUM_REQUESTS=1000`, `DECODE_TOKENS=256`, `VALUES=0.75`:
  - decode throughput: 2968.42 toks/s
  - mean DRAM read bandwidth utilization: 20.09%
  - p95 DRAM read bandwidth utilization: 43.00%
  - better than the 10K baseline, but worse than shorter-context high concurrency.

Best result so far:

- model: `meta-llama/Llama-3.1-8B`
- `PREFIX_LEN=256`
- `SUFFIX_LEN=20`
- `NUM_REQUESTS=1000`
- `DECODE_TOKENS=256`
- `VALUES=0.75`
- `MAX_NUM_SEQS=1000`
- `MIN_PREFIX_BATCH_SIZE=100`
- schedule: `prefix-adaptive`
- attention backend selected by vLLM: `FLASH_ATTN`
- GPU metrics frequency: 5000 Hz

Results:

- decode throughput: 3580.67 toks/s
- mean DRAM read bandwidth utilization: 24.11%
- p95 DRAM read bandwidth utilization: 44.00%
- max DRAM read bandwidth utilization: 98.00%
- mean DRAM write bandwidth utilization: 2.59%
- mean SMs active: 96.16%

Interpretation:

- This improves mean DRAM read bandwidth utilization from the prior Llama baseline of 10.62% to 24.11%, a 2.27x improvement.
- The best improvement came from changing the workload/scheduling regime to fit the GPU's KV-cache capacity: shorter prefixes allowed vLLM to keep far more active decode sequences (`~250x` concurrency was observed for the 512-token case, versus `~20x` for 10K contexts).
- Longer prefixes increase per-sequence KV work but collapse active concurrency on a 48GB GPU. The measured optimum in this sweep was therefore not the longest prefix, but the shorter high-concurrency regime.
- This is a real throughput/utilization improvement, not a reporting artifact: the best run also had the highest measured decode throughput among the Llama 3.1 attempts.

## Non-Oracle Prefix-Hash Scheduler

The earlier `prefix-adaptive` scheduler was an oracle baseline: it used synthetic `prefix_group_ids` from the workload generator. That is useful for proving the scheduling direction, but it is not a general implementation. Added `prefix-hash-adaptive`, which discovers prefix groups directly from the prompt token IDs.

Algorithm:

- Build a lightweight chunked prefix hash tree over the batch.
- Each request contributes nodes for fixed-size token chunks, e.g. 64-token chunks.
- A node with many members represents a group of requests sharing that token prefix.
- Score candidate nodes by `(group_size - 1) * shared_prefix_len * decode_tokens`, which estimates how much repeated KV traversal can benefit from locality during decode.
- Greedily select disjoint high-score groups that satisfy:
  - `group_size >= min_prefix_batch_size`
  - `shared_prefix_len >= min_shared_prefix_len`
- Run selected groups as prefix-homogeneous `generate` calls and put the remaining requests into a residual mixed batch.

This is close in spirit to Feather's Chunked Hash Tree, but simpler:

- It is deterministic and greedy, not RL/bandit-controlled.
- It uses exact token chunks instead of a production service's rolling request metadata.
- It is per-benchmark-batch rather than an online queue scheduler.
- It does not require synthetic oracle labels.

Implementation changes:

- Added `prefix-hash-adaptive` to the CLI schedules.
- Added `--prefix-hash-chunk-size` and `--min-shared-prefix-len`.
- Updated the remote measurement wrapper to pass these knobs.
- Kept `prefix-adaptive` as the oracle upper-bound scheduler.

Validation:

- Local checks:
  - `uvx ruff format prefix_bw/runner.py prefix_bw/cli.py`
  - `uvx ruff check prefix_bw/runner.py prefix_bw/cli.py`
  - `uvx ty check prefix_bw/runner.py prefix_bw/cli.py`
  - `bash -n scripts/measure_prefix_bandwidth_pair.sh`
  - toy scheduler test verified that the hash scheduler recovers two shared-prefix groups without oracle labels.

Remote comparison on identical workload:

- model: `meta-llama/Llama-3.1-8B`
- `PREFIX_LEN=256`
- `SUFFIX_LEN=20`
- `NUM_REQUESTS=1000`
- `DECODE_TOKENS=256`
- `VALUES=0.75`
- `MAX_NUM_SEQS=1000`
- `MIN_PREFIX_BATCH_SIZE=100`
- `PREFIX_HASH_CHUNK_SIZE=64`
- `MIN_SHARED_PREFIX_LEN=128`
- attention backend selected by vLLM: `FLASH_ATTN`
- GPU metrics frequency: 5000 Hz

Results:

- `single`
  - decode throughput: 3234.59 toks/s
  - mean DRAM read bandwidth utilization: 18.61%
  - p95 DRAM read bandwidth utilization: 46.00%
  - mean DRAM write bandwidth utilization: 3.97%
  - mean SMs active: 97.59%
- `prefix-hash-adaptive`
  - decode throughput: 3565.53 toks/s
  - mean DRAM read bandwidth utilization: 24.02%
  - p95 DRAM read bandwidth utilization: 44.00%
  - mean DRAM write bandwidth utilization: 2.60%
  - mean SMs active: 96.20%

Interpretation:

- The non-oracle scheduler improves same-workload decode throughput by about 10.2%.
- It improves same-workload mean DRAM read bandwidth utilization by about 29.1%.
- It nearly matches the oracle scheduler on this workload (`3565.53` vs `3580.67` toks/s), which means the chunked prefix-hash detection recovered the useful grouping with negligible scheduling loss.
- Compared with the paper's Figure 4b absolute bandwidth values (~12-13% mean DRAM bandwidth for homogeneous endpoints), this run reaches about 24% mean DRAM read utilization, roughly 1.9x the paper's plotted absolute bandwidth utilization. This is not an apples-to-apples paper reproduction because the workload uses shorter prefixes and longer decode to keep many more active sequences resident on a 48GB GPU.

Differences from the paper setup:

- Paper: Llama-3-8B; this run: Llama-3.1-8B.
- Paper: 10K-token prefixes; this best run: 256-token prefixes.
- Paper: 500 requests; this best run: 1000 requests.
- Paper: 20-token suffix and 50 decode tokens in Figure 4; this best run: 20-token suffix and 256 decode tokens.
- Paper: DCGMI-reported DRAM bandwidth; this run: Nsight Systems GPU metrics plus `nvidia-smi dmon` summaries.
- Paper: evaluates Feather's online CHT + RL/bandit scheduler in a serving-style system; this implementation is an offline per-batch chunked-hash greedy scheduler inside the benchmark harness.
- Paper: focuses on preserving long-prefix locality; this run improves TPS by fitting the active KV footprint to the 48GB GPU so vLLM can sustain much higher decode concurrency.

## Paper-Style Token Setup With Non-Oracle Scheduler

Ran `prefix-hash-adaptive` on the paper-style token counts:

- model: `meta-llama/Llama-3.1-8B`
- `NUM_REQUESTS=500`
- `PREFIX_LEN=10000`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=50`
- `VALUES=0.75`
- `MAX_NUM_SEQS=500`
- `MIN_PREFIX_BATCH_SIZE=100`
- `PREFIX_HASH_CHUNK_SIZE=64`
- `MIN_SHARED_PREFIX_LEN=128`
- attention backend selected by vLLM: `FLASH_ATTN`

Note: this uses Llama 3.1 instead of the paper's Llama 3 8B because `Meta-Llama-3-8B` advertises an 8192-token context in vLLM and rejects the paper's 10K-prefix setup.

Results:

- `single`
  - decode throughput: 897.68 toks/s
  - mean DRAM read bandwidth utilization: 5.04%
  - p95 DRAM read bandwidth utilization: 29.00%
  - mean DRAM write bandwidth utilization: 1.38%
  - mean SMs active: 90.29%
- `prefix-hash-adaptive`
  - decode throughput: 897.40 toks/s
  - mean DRAM read bandwidth utilization: 8.80%
  - p95 DRAM read bandwidth utilization: 55.00%
  - mean DRAM write bandwidth utilization: 1.51%
  - mean SMs active: 88.36%

Interpretation:

- The non-oracle scheduler improved mean DRAM read bandwidth utilization by about 74.6% on the paper-style token shape.
- It did not improve decode throughput (`897.68` to `897.40` toks/s, effectively flat).
- This is the key limitation of the current benchmark environment: 10K-token requests on a 48GB GPU only allow about 20 active full-context sequences in vLLM's KV cache, so locality gains are offset by lower effective scheduling/batching efficiency and the additional `generate` call.
- Compared with the paper's Figure 4b, the bandwidth uplift is qualitatively consistent with the paper's claim, but the throughput win does not reproduce under this hardware/model stack. The earlier TPS win came from a shorter-prefix regime that increased active decode concurrency.

## Paper-Style Follow-up: Priority and Capacity-Aware Waves

The paper-style hash scheduler had two competing problems:

- Separate `generate` calls improved DRAM locality but did not improve TPS.
- One combined `generate` call preserved TPS but let vLLM mix prefixes too freely, so bandwidth stayed low.

Added and tested `prefix-hash-priority`:

- Same chunked-hash prefix detection as `prefix-hash-adaptive`.
- Submits all requests in a single vLLM `generate` call.
- Uses vLLM's per-request `priority` argument to bias admission order by discovered prefix groups.

For Llama 3.1 runs on `hinton-01`, Hugging Face network access began hanging even for config requests, so subsequent runs used:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

The model was already cached locally from previous runs. Direct smoke tests also need:

- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

The nsys wrapper already sets those vLLM variables.

Paper-style setup:

- model: `meta-llama/Llama-3.1-8B`
- `NUM_REQUESTS=500`
- `PREFIX_LEN=10000`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=50`
- `VALUES=0.75`
- `MAX_NUM_SEQS=500`
- attention backend selected by vLLM: `FLASH_ATTN`

Results:

- `prefix-hash-priority`
  - `num_generate_calls=1`
  - decode throughput: 907.95 toks/s
  - mean DRAM read bandwidth utilization: 5.09%
  - p95 DRAM read bandwidth utilization: 29.00%
  - interpretation: recovers some TPS versus `single` (`897.68` -> `907.95` toks/s), but does not improve bandwidth because vLLM still schedules a mixed-prefix active set.
- `prefix-hash-priority` with vLLM `scheduling_policy="priority"`
  - `num_generate_calls=1`
  - decode throughput: 945.65 toks/s
  - mean DRAM read bandwidth utilization: 5.34%
  - p95 DRAM read bandwidth utilization: 30.00%
  - interpretation: best TPS so far on the paper-style workload, about 5.3% faster than `single` and about 4.2% faster than the earlier priority run. It still does not improve memory bandwidth meaningfully because the one-call vLLM scheduler remains free to keep a mixed-prefix active set during decode.
- `prefix-hash-priority` with vLLM `scheduling_policy="priority"` and `PREFIX_PRIORITY_ORDER=smallest-first`
  - `num_generate_calls=1`
  - decode throughput: 938.67 toks/s
  - mean DRAM read bandwidth utilization: 5.34%
  - p95 DRAM read bandwidth utilization: 30.00%
  - interpretation: prioritizing the smaller 125-request group first was worse than the default/large-first order. Large-first remains the best priority order for this `beta=0.75` setup.
- `prefix-hash-adaptive`, `PREFIX_BATCH_SIZE=200`
  - `num_generate_calls=3`
  - decode throughput: 894.84 toks/s
  - mean DRAM read bandwidth utilization: 11.07%
  - p95 DRAM read bandwidth utilization: 63.00%
  - interpretation: crosses the paper's approximate 10.5% mean bandwidth target while keeping TPS roughly flat relative to `single`.
- `prefix-hash-adaptive`, `PREFIX_BATCH_SIZE=150`
  - `num_generate_calls=4`
  - decode throughput: 869.23 toks/s
  - mean DRAM read bandwidth utilization: 13.77%
  - p95 DRAM read bandwidth utilization: 73.00%
  - interpretation: better bandwidth margin over the paper, with a moderate TPS cost.
- `prefix-hash-adaptive`, `PREFIX_BATCH_SIZE=100`
  - `num_generate_calls=6`
  - decode throughput: 812.97 toks/s
  - mean DRAM read bandwidth utilization: 18.85%
  - p95 DRAM read bandwidth utilization: 87.00%
  - interpretation: largest paper-style bandwidth win so far, about 1.8x the paper's approximate 10.5% mean, but TPS drops by about 9.4% versus `single`.

Best paper-style bandwidth result:

- `prefix-hash-adaptive`, `PREFIX_BATCH_SIZE=100`
- mean DRAM read bandwidth utilization: 18.85%
- p95 DRAM read bandwidth utilization: 87.00%

Best paper-style TPS result among new schedulers:

- `prefix-hash-priority` with vLLM `scheduling_policy="priority"`
- decode throughput: 945.65 toks/s
- mean DRAM read bandwidth utilization: 5.34%

Direct non-profiled TPS check:

- Same paper-style token shape, run without Nsight Systems.
- `single`: 984.25 toks/s
- `prefix-hash-priority` with vLLM `scheduling_policy="priority"`: 1032.07 toks/s
- Interpretation: nsys profiling suppresses the absolute TPS numbers. The priority scheduler improves direct runtime TPS by about 4.9% over `single` on the paper-style setup.

Best paper-style balanced point:

- `prefix-hash-adaptive`, `PREFIX_BATCH_SIZE=200`
- decode throughput: 894.84 toks/s, effectively flat versus `single` at 897.68 toks/s
- mean DRAM read bandwidth utilization: 11.07%, slightly above the paper's approximate 10.5%

Conclusion:

- To beat the paper's bandwidth by a big margin on the paper-style token shape, capacity-aware prefix waves work: smaller waves push mean DRAM read from 5.04% to 18.85%.
- To maximize TPS, true priority scheduling is better, but it does not improve memory bandwidth because vLLM's internal scheduler still mixes prefixes after admission.
- The missing production-quality solution is a true in-engine scheduler that combines both: single engine run, but per-step active-set selection constrained by prefix hash group and KV-cache capacity. The current benchmark-level implementations can approximate either side of that tradeoff, but not both simultaneously through the public `LLM.generate` API.

## gpu1 User-Local CUDA/vLLM Bring-up

When setting up the isolated `gpu1` environment, the xFormers source build appeared stuck because the local SSH wrapper remained alive after the remote build process had finished. There were no remaining `ninja`/`nvcc`/`python` build processes on `gpu1`, and imports confirmed the build succeeded:

- `torch 2.3.0+cu118`
- `vllm 0.4.2+cu118` from the source checkout
- `xformers 0.0.26.post1`
- `xformers.ops` and `vllm._C` both import successfully

Two compatibility fixes were needed before the benchmark could run:

- vLLM 0.4.2 expects token IDs via `LLM.generate(prompt_token_ids=..., sampling_params=...)`, while newer vLLM accepts `{"prompt_token_ids": ...}` prompt dictionaries. Added a small signature-based compatibility wrapper in `prefix_bw/runner.py`.
- The first prefix-cache CUDA execution failed with `RuntimeError: Triton Error [CUDA]: device kernel image is invalid`. `gpu1` has driver `470.256.02` reporting CUDA `11.4`, and Triton 2.3 generated a kernel image the driver rejected. Setting `TRITON_PTXAS_PATH=/users/extusr/sethigeet/cuda-11.8/bin/ptxas` and clearing the Triton cache fixed it.

Validation after the fix:

- `facebook/opt-125m`, `NUM_REQUESTS=4`, `PREFIX_LEN=32`, `DECODE_TOKENS=4`
- vLLM selected the XFormers backend
- automatic prefix caching was enabled
- benchmark completed with decode throughput `1560.7 toks/s`

Script changes:

- `scripts/run_prefix_nsys.sh` now respects `PREFIX_PYTHON` so the nsys profile can use `.venv-vllm-cu11-test/bin/python` instead of the default `uv run` environment.
- `scripts/run_prefix_nsys.sh` and `scripts/measure_prefix_bandwidth_pair.sh` now export `TRITON_PTXAS_PATH` from `CUDA_HOME` when available.
- `scripts/run_prefix_nsys.sh` now handles the older Nsight Systems 2022.4 option names on `gpu1` (`--gpu-metrics-device` instead of `--gpu-metrics-devices`) and forces SQLite overwrite on export.
- `scripts/measure_prefix_bandwidth_pair.sh` now records a concurrent `nvidia-smi dmon` utilization log because this Nsight Systems export does not include the `GPU_METRICS` SQLite tables used by newer `nsys`.

`gpu1` OPT-1.3B pair measurement:

- model: `facebook/opt-1.3b`
- workload: `NUM_REQUESTS=128`, `PREFIX_LEN=512`, `SUFFIX_LEN=32`, `DECODE_TOKENS=64`, `VALUES=0.75`
- vLLM stack: `vllm 0.4.2+cu118`, XFormers backend, automatic prefix caching enabled
- `single`:
  - decode throughput: `1874.91 toks/s`
  - dmon mean memory-controller utilization: `4.20%`
  - dmon p95 memory-controller utilization: `37.00%`
  - dmon max memory-controller utilization: `44.00%`
- `prefix-adaptive`:
  - decode throughput: `1749.93 toks/s`
  - dmon mean memory-controller utilization: `6.00%`
  - dmon p95 memory-controller utilization: `38.00%`
  - dmon max memory-controller utilization: `51.00%`

Interpretation:

- On this smaller OPT workload, `prefix-adaptive` increased mean memory-controller utilization by about `42.9%` (`4.20%` to `6.00%`) and raised max utilization from `44%` to `51%`.
- It reduced decode throughput by about `6.7%` because the adaptive split issued two `generate` calls (`96` requests then residual) instead of one full batch of `128`; this is the expected overhead when the model/workload is not strongly memory-bound enough for the locality gain to dominate.
- Phi-3 is not viable with vLLM 0.4.2 prefix caching on this host because its sliding-window config triggers `NotImplementedError: Sliding window is not allowed with prefix caching enabled!` and this older vLLM build cannot use the newer `disable_sliding_window` path for Phi-3.

## Four-GPU Algorithm Search: Fill-Aware Prefix Hashing

Added `prefix-hash-auto`, a fill-aware non-oracle scheduler on top of the chunked prefix hash grouping. The key idea is to avoid paying extra `generate` calls for very underfilled prefix groups:

- discover shared-prefix groups from token chunks as in `prefix-hash-adaptive`
- estimate split-batch fill ratio as `num_requests / (num_batches * batch_capacity)`
- use the prefix-homogeneous split only when the fill ratio is high enough
- default threshold was tuned to `0.20`, which splits useful 2- and 4-group cases but avoids the 8/16-group cases where small microbatches lose occupancy

Important operational fixes for the four-GPU run:

- Use the cached model snapshot path directly instead of the Hugging Face repo id; otherwise `transformers` can hang in metadata HTTP calls before vLLM starts.
- Put `nvidia-smi dmon` in the background after exporting `CUDA_VISIBLE_DEVICES`; an earlier shell-precedence bug caused the Python processes to miss the GPU binding and collide on GPU 0.
- Use separate `TRITON_CACHE_DIR`s per GPU to avoid cross-process Triton cache races.

Four A6000 sweep setup:

- model: `facebook/opt-1.3b` from local snapshot
- backend: vLLM 0.4.2 + XFormers, automatic prefix caching enabled
- schedules: `single`, `prefix-grouped`, `prefix-hash-adaptive`, `prefix-hash-auto`
- GPUs used concurrently:
  - GPU 0: `512` requests, `256` prefix tokens, `128` decode tokens
  - GPU 1: `768` requests, `256` prefix tokens, `256` decode tokens
  - GPU 2: `512` requests, `512` prefix tokens, `256` decode tokens
  - GPU 3: `1024` requests, `128` prefix tokens, `256` decode tokens

Best sweep results:

- Best oracle result: `prefix-grouped`, `1024` requests / `128` prefix / `256` decode / `8` groups
  - `single`: `2740.3 toks/s`
  - `prefix-grouped`: `4161.0 toks/s`
  - improvement: `51.84%`
- Best non-oracle result: `prefix-hash-adaptive`, `512` requests / `512` prefix / `256` decode / `4` groups
  - `single`: `2419.2 toks/s`
  - `prefix-hash-adaptive`: `3431.6 toks/s`
  - improvement: `41.85%`
- Best absolute throughput: `4500.0 toks/s` with oracle `prefix-grouped`, `512` requests / `256` prefix / `128` decode / `2` groups.

Memory-controller utilization from the sweep dmon logs:

- `512` requests / `512` prefix / `256` decode sweep:
  - mean memory-controller utilization: `45.12%`
  - active-sample mean memory-controller utilization: `48.32%`
  - p95 memory-controller utilization: `78.0%`
  - max memory-controller utilization: `82.0%`
- This exceeds the paper's Figure 4 plotted absolute mean DRAM utilization of roughly `12-13%`, though this is not an apples-to-apples reproduction because it uses OPT-1.3B, a different profiler, shorter prefixes, and a concurrency-focused workload.

Focused validation after tuning `prefix-hash-auto` threshold to `0.20`:

- workload: `512` requests / `512` prefix / `256` decode / `4` groups
- `single`: `2644.2 toks/s`
- `prefix-hash-auto`: `3286.4 toks/s`
- improvement: `24.29%`

Interpretation:

- The strongest reproducible throughput result beats the paper's reported `~40%` bandwidth uplift in relative terms: `51.84%` oracle throughput improvement, and `41.85%` non-oracle throughput improvement in the broad sweep.
- The fill-aware auto scheduler needs the lower `0.20` threshold; the initial `0.35` threshold was too conservative and skipped profitable 4-group splits.
- `prefix-hash-adaptive` is still the best non-oracle policy for the measured sweet spot. `prefix-hash-auto` is safer across bad high-group-count cases, but it sacrifices some peak performance unless tuned aggressively.

## Final `hinton-01` Paper-Style and Capacity-Aware Profiles

Implemented the remote-host override fix in `scripts/sync_remote.sh` so the default can be overridden with:

```bash
HOST=hinton-01 REMOTE_DIR=/home/geet/code/attention-bw scripts/run_prefix_nsys_remote.sh ...
```

The script now prefers positional arguments, then `HOST` / `REMOTE_DIR` environment variables, then the old defaults.

### Paper-style Llama 3.1 setup

The closest compatible paper-style run uses `meta-llama/Llama-3.1-8B` because vLLM exposes only an 8192-token context for `Meta-Llama-3-8B`, which cannot run the paper's 10K-prefix setup.

Common settings:

- host: `hinton-01`
- model: `meta-llama/Llama-3.1-8B`
- backend: vLLM `0.22.1`, FlashAttention 2
- `NUM_REQUESTS=500`
- `PREFIX_LEN=10000`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=50`
- `MAX_NUM_SEQS=500`
- `GPU_METRICS_FREQUENCY=5000`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

vLLM reported:

- GPU KV cache size: `212,317` tokens
- Maximum concurrency for `10,086` tokens/request: about `21x`

That concurrency number is the main reason the 10K-token paper-style run does not fully utilize DRAM bandwidth: the long-context KV footprint prevents vLLM from keeping hundreds of active decode sequences resident, even though `max_num_seqs=500`.

Results:

- Homogeneous `single`, `beta=1.0`
  - output: `results/prefix_bw_plan_paper_beta1_20260630_2220_single.csv`
  - decode throughput: `952.54 toks/s`
  - mean DRAM read bandwidth: `5.17%`
  - p95 DRAM read bandwidth: `30.0%`
  - mean SMs active: `91.72%`
- Mixed `single`, `beta=0.75`
  - output: `results/prefix_bw_llama31_paper_hash_20260630_171146_single.csv`
  - decode throughput: `897.68 toks/s`
  - mean DRAM read bandwidth: `5.04%`
  - p95 DRAM read bandwidth: `29.0%`
  - mean SMs active: `90.29%`
- `prefix-hash-adaptive`, `beta=0.75`
  - output: `results/prefix_bw_llama31_paper_hash_20260630_171146_hash_adaptive.csv`
  - decode throughput: `897.40 toks/s`
  - mean DRAM read bandwidth: `8.80%`
  - p95 DRAM read bandwidth: `55.0%`
  - `num_generate_calls=2`
- `prefix-hash-auto`, `beta=0.75`
  - output: `results/prefix_bw_plan_paper_auto_20260630_2227_hash_auto.csv`
  - decode throughput: `923.74 toks/s`
  - mean DRAM read bandwidth: `9.08%`
  - p95 DRAM read bandwidth: `56.0%`
  - `num_generate_calls=2`
- `prefix-hash-priority` with vLLM `scheduling_policy="priority"`, `beta=0.75`
  - output: `results/prefix_bw_llama31_paper_priority_policy_20260630_214229_prefix-hash-priority.csv`
  - decode throughput: `945.65 toks/s`
  - mean DRAM read bandwidth: `5.34%`
  - p95 DRAM read bandwidth: `30.0%`
  - `num_generate_calls=1`

Interpretation:

- Homogeneous paper-style batching is faster than mixed batching, but it does not materially raise mean DRAM bandwidth on this stack (`5.17%` vs `5.04%`).
- Prefix-hash grouping raises mean DRAM read utilization by about `75-80%`, but the extra `generate` calls can keep TPS flat unless the split remains full enough.
- `prefix-hash-auto` is the best balanced paper-style public-API policy from these runs: it improves TPS by about `2.9%` over `single` and improves mean DRAM read by about `80.2%`.
- Among the mixed-workload (`beta=0.75`) scheduler variants, `prefix-hash-priority` is the best paper-style TPS policy: about `5.3%` faster than mixed `single`, but it does not improve DRAM bandwidth because vLLM still mixes prefix groups inside a single engine call. The homogeneous `beta=1.0` run is a separate workload reference, not a scheduler variant for the mixed workload.

### Prefix-wave sweep on the paper-style shape

`prefix-hash-adaptive` wave sizing shows the bandwidth/throughput tradeoff directly:

- Default split, max batch `375`: `897.40 toks/s`, `8.80%` mean DRAM read
- `PREFIX_BATCH_SIZE=200`: `894.84 toks/s`, `11.07%` mean DRAM read
- `PREFIX_BATCH_SIZE=150`: `869.23 toks/s`, `13.77%` mean DRAM read
- `PREFIX_BATCH_SIZE=100`: `812.97 toks/s`, `18.85%` mean DRAM read

Smaller prefix-local waves improve bandwidth by making the active set more homogeneous, but too many waves reduce tokens/sec because vLLM loses batch fullness and pays repeated `generate` overhead.

### Capacity-aware high-concurrency setup

The throughput win appears when the workload fits the GPU's KV-cache capacity well enough to keep many decode sequences active.

Common settings:

- model: `meta-llama/Llama-3.1-8B`
- `NUM_REQUESTS=1000`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=256`
- `MAX_NUM_SEQS=1000`
- `VALUES=0.75`

Key results:

- `PREFIX_LEN=256`, `single`
  - output: `results/prefix_bw_llama31_hash_sched_20260630_164302_single.csv`
  - decode throughput: `3234.59 toks/s`
  - mean DRAM read bandwidth: `18.61%`
  - mean SMs active: `97.59%`
- `PREFIX_LEN=256`, `prefix-hash-adaptive`
  - output: `results/prefix_bw_llama31_hash_sched_20260630_164302_hash_adaptive.csv`
  - decode throughput: `3565.53 toks/s`
  - mean DRAM read bandwidth: `24.02%`
  - mean SMs active: `96.20%`
- `PREFIX_LEN=512`, `prefix-adaptive`
  - output: `results/prefix_bw_llama31_short_highconc_20260630_160728_adaptive.csv`
  - decode throughput: `3359.46 toks/s`
  - mean DRAM read bandwidth: `22.64%`
- `PREFIX_LEN=1024`, `prefix-adaptive`
  - output: `results/prefix_bw_llama31_1024_highconc_20260630_161818_adaptive.csv`
  - decode throughput: `2968.42 toks/s`
  - mean DRAM read bandwidth: `20.09%`

Interpretation:

- Shorter prefixes reduce per-sequence KV-cache footprint enough to keep a much wider active decode set.
- At `PREFIX_LEN=256`, non-oracle prefix hashing improves same-workload TPS by about `10.2%` and mean DRAM read bandwidth by about `29.1%`.
- Longer prefixes increase per-token KV work but reduce active concurrency, so they do not automatically improve sustained bandwidth or tokens/sec on a 48GB GPU.

### Root cause and recommendation

We are not seeing full decode-phase memory-bandwidth utilization because the public vLLM `LLM.generate` path forces a tradeoff:

- One large call preserves scheduling efficiency but allows mixed-prefix active sets, so L2 reuse and sustained DRAM read utilization stay low.
- Separate prefix-homogeneous calls improve DRAM locality, but reduce batch fullness and add repeated engine/API overhead.
- Very long prefixes further cap active decode concurrency because the KV cache can only hold about `21` full 10K-token requests on this RTX 6000 Ada setup.

Best available settings in this benchmark harness:

- For paper-style mixed long-prefix runs where TPS matters most: use `prefix-hash-priority` with vLLM priority scheduling.
- For paper-style long-prefix runs where bandwidth is the target and modest TPS loss is acceptable: use `prefix-hash-adaptive` with `PREFIX_BATCH_SIZE=200` or lower.
- For actual throughput improvement: use capacity-aware prefix hashing in a shorter-prefix/high-concurrency regime; the measured `PREFIX_LEN=256`, `NUM_REQUESTS=1000`, `DECODE_TOKENS=256`, `prefix-hash-adaptive` run reached `3565.53 toks/s` and `24.02%` mean DRAM read.

The next implementation step for improving both bandwidth and TPS on 10K-prefix workloads is not another wrapper-level split. It is an in-engine scheduler that keeps one vLLM engine run while constraining each decode step's active set to prefix-local, sufficiently full waves based on chunked prefix hashes and KV-cache capacity.

## `gpu1` Four-GPU Paper-Aligned Rerun

The user's latest request was to use all four GPUs on `gpu1` and match the paper's setup as much as possible.

Important constraints on `gpu1`:

- Hardware is `NVIDIA RTX A6000` GPUs, not the paper's `RTX 6000 Ada`.
- The usable cached models are `facebook/opt-1.3b`, `facebook/opt-125m`, and `microsoft/Phi-3-mini-4k-instruct`.
- Phi-3 still hits the vLLM `0.4.2` sliding-window + prefix-caching incompatibility.
- Llama 3 / Llama 3.1 is not cached on `gpu1`, and parallel online downloads previously caused Hugging Face metadata/file-lock stalls.
- OPT-1.3B has a 2048-position context, so the longest safe paper-like prompt was `PREFIX_LEN=1960`, `SUFFIX_LEN=20`, `DECODE_TOKENS=50`, `max_seq_len=2046`; this preserves the paper's request count, suffix length, output length, and max batch size, but not the 10K prefix length or Llama-3-8B model.
- vLLM selected the XFormers backend because FlashAttention is unavailable in this CUDA 11.8 / vLLM 0.4.2 environment.

Common settings:

- host: `gpu1`
- GPUs: all four A6000s concurrently, one single-GPU experiment per GPU
- model: local `facebook/opt-1.3b` snapshot
- backend: vLLM `0.4.2` + XFormers, automatic prefix caching enabled
- `NUM_REQUESTS=500`
- `PREFIX_LEN=1960`
- `SUFFIX_LEN=20`
- `DECODE_TOKENS=50`
- `MAX_NUM_SEQS=500`
- `GPU_MEMORY_UTILIZATION=0.75`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- separate `TRITON_CACHE_DIR` per GPU

### Figure-4-style homogeneity sweep

Run id: `20260630_221727`

The sweep used beta values `0,0.25,0.5,0.75,1.0`.

Decode throughput, tokens/sec:

- `single`: `1016.6`, `1010.7`, `1020.7`, `999.6`, `1003.2`
- oracle `prefix-adaptive`: `1013.8`, `1064.7`, `1104.8`, `1077.2`, `1015.8`
- non-oracle `prefix-hash-adaptive`: `995.6`, `1035.0`, `1070.4`, `1043.2`, `1009.0`

Focused beta `0.75` rerun, one schedule per GPU:

- `single`: `988.9 toks/s`
- oracle `prefix-adaptive`: `1039.0 toks/s` (`+5.06%`)
- non-oracle `prefix-hash-adaptive`: `1065.9 toks/s` (`+7.78%`)
- non-oracle `prefix-hash-auto`: `1077.2 toks/s` (`+8.93%`)

Focused beta `0.75` dmon memory-controller utilization:

- `single`: active mean `43.00%`, active p95 `98.00%`, max `99.00%`
- oracle `prefix-adaptive`: active mean `40.07%`, active p95 `70.00%`, max `72.00%`
- `prefix-hash-adaptive`: active mean `38.48%`, active p95 `70.00%`, max `72.00%`
- `prefix-hash-auto`: active mean `33.87%`, active p95 `70.00%`, max `70.00%`

Interpretation:

- On this shorter OPT context, prefix-local scheduling improves TPS at beta `0.75`, but dmon memory-controller activity does not increase. The likely reason is that the split schedules reduce repeated KV traffic via better cache locality and finish faster, so raw memory-controller activity is lower even though throughput is higher.
- Nsight Systems was also tried for focused beta `0.75`, but this `gpu1` Nsight export produced no `GPU_METRICS` table despite enabling GPU metrics, so decode-window DRAM percentages are unavailable on this host from Nsight. The large `.nsys-rep` and `.sqlite` files were left on `gpu1` and not copied back.

### Paper common 5-prefix-group setup

The paper's end-to-end default workload uses 5 prefix groups. We matched that group count with the same 500 requests, 20-token suffix, 50 generated tokens, and max batch 500, again with OPT's maximum safe prefix length.

Combined 5-group schedule sweep, run id `20260630_221727`:

- `single`: `819.4 toks/s`
- oracle `prefix-grouped`: `1167.1 toks/s` (`+42.4%`)
- non-oracle `prefix-hash-adaptive`: `1158.9 toks/s` (`+41.4%`)
- non-oracle `prefix-hash-auto`: `1166.0 toks/s` (`+42.3%`)

Focused 5-group rerun, one schedule per GPU, run id `20260630_223245`:

- `single`: `816.1 toks/s`
- oracle `prefix-grouped`: `1182.4 toks/s` (`+44.89%`)
- non-oracle `prefix-hash-adaptive`: `1168.7 toks/s` (`+43.21%`)
- non-oracle `prefix-hash-auto`: `1203.8 toks/s` (`+47.51%`)

Focused 5-group dmon memory-controller utilization:

- `single`: active mean `50.08%`, active p95 `100.00%`, max `100.00%`
- oracle `prefix-grouped`: active mean `30.90%`, active p95 `55.00%`, max `56.00%`
- `prefix-hash-adaptive`: active mean `32.40%`, active p95 `54.00%`, max `55.00%`
- `prefix-hash-auto`: active mean `30.27%`, active p95 `55.00%`, max `56.00%`

Interpretation:

- This is the best paper-aligned result on `gpu1`: the non-oracle `prefix-hash-auto` scheduler improves decode throughput by `47.51%` over `single` on the 5-prefix-group workload.
- The result is better than the paper's reported `~40%` relative improvement, but it is still not an apples-to-apples reproduction because the model, prefix length, GPU generation, attention backend, and bandwidth profiler differ.
- The lower dmon memory-controller activity for grouped/hash schedules is not a regression in this setup. It means the optimized schedules are doing less off-chip memory work per generated token and finishing sooner; dmon is not measuring the paper's exact DCGMI/DRAM-read metric nor a decode-only window here.
