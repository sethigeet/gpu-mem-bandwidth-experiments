# Prefix BW Memory Investigation

## Goal

Investigate whether vLLM decode throughput improves when requests that share long
prefixes are decoded in prefix-local groups. The motivation comes from Feather's
cache-aware batching result: mixed-prefix decode batches repeatedly read unrelated
KV-cache regions, while prefix-homogeneous batches can reuse cache lines more
effectively.

## Maintained Schedulers

- `single`: the baseline. Submit all requests in one `llm.generate` call and let
  vLLM schedule the batch normally.
- `prefix-grouped`: oracle baseline. Use synthetic workload `prefix_group_ids` to
  run each known same-prefix group as one or more microbatches.
- `prefix-adaptive`: oracle baseline with a guardrail. Only split same-prefix
  groups with at least `--min-prefix-batch-size` requests; leave smaller groups
  in a residual mixed batch.
- `prefix-hash-adaptive`: non-oracle scheduler. Build a lightweight Chunked Hash
  Tree analogue from token IDs, choose disjoint high-benefit shared-prefix groups,
  and run each group as prefix-local waves.
- `prefix-hash-auto`: same prefix-hash grouping, but fall back to `single` if the
  planned waves are too underfilled.
- `offline-prefix-wave`: conservative prefix-hash planner. It estimates saved
  shared-prefix decode work and only splits when the estimated gain clears fill
  and net-gain thresholds.

Patch-only in-engine experiments and direct engine driving were removed from the
maintained code path. They were useful for exploration, but they depend on local
vLLM internals and are not stable benchmark features.

## How Prefix Hashing Works

Each prompt is split into fixed-size token chunks, defaulting to 64 tokens. The
planner builds tree nodes keyed by `(parent_node, chunk_tokens)`. A deep node with
many request members represents a long shared prefix. Candidate nodes must satisfy
both `--min-prefix-batch-size` and `--min-shared-prefix-len`.

Candidates are sorted by estimated saved work:

```text
(num_requests_in_group - 1) * shared_prefix_len * decode_tokens
```

The selection is greedy. The highest-scoring candidate claims its requests first;
later candidates are reduced to unclaimed requests and skipped if they become too
small. This keeps the final groups disjoint and favors deeper, larger shared
prefixes.

## Key Results

10K-prefix paper-style runs on Llama 3.1 showed the central tradeoff:

- `single`: `897.68 toks/s`, mean DRAM read `5.04%`.
- `prefix-hash-adaptive`: `897.40 toks/s`, mean DRAM read `8.80%`.
- `prefix-hash-auto`: `923.74 toks/s`, mean DRAM read `9.08%`.
- `prefix-hash-adaptive` with `--prefix-batch-size 100`: `812.97 toks/s`,
  mean DRAM read `18.85%`, p95 DRAM read `87.00%`.

Smaller prefix-local waves improved bandwidth by making the active set more
homogeneous, but too many waves reduced tokens/sec because the run lost batch
fullness and paid repeated `generate` overhead.

High-concurrency shorter-prefix runs showed where the non-oracle scheduler can
improve both throughput and memory utilization:

- `single`: `3234.59 toks/s`, mean DRAM read `18.61%`.
- `prefix-hash-adaptive`: `3565.53 toks/s`, mean DRAM read `24.02%`.

The best broad-sweep non-oracle result was `prefix-hash-adaptive` with 512
requests, 512-token prefixes, 256 decode tokens, and 4 prefix groups:

- `single`: `2419.2 toks/s`.
- `prefix-hash-adaptive`: `3431.6 toks/s`.

## Takeaways

Prefix locality is real, but bandwidth alone is not the objective. On long
10K-prefix workloads, a 48GB GPU can keep only a limited number of full-context
sequences active, so extra waves can erase the locality gain. On workloads that
fit the KV-cache capacity better, prefix-hash scheduling can improve both decode
throughput and measured DRAM read utilization.

The maintained code therefore keeps conservative schedulers and guardrails:
oracle baselines for controlled comparison, non-oracle prefix hashing for
realistic grouping, and `offline-prefix-wave` / `prefix-hash-auto` fallbacks for
underfilled cases.

## Next Steps

The next production-quality step would be an in-engine scheduler that preserves a
single vLLM run while choosing prefix-local active decode waves at each scheduler
step. Wrapper-level splitting is useful for benchmarking the effect, but it
cannot fully combine prefix locality with vLLM's batching efficiency.
