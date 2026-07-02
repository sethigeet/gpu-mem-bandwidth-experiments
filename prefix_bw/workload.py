"""Synthetic prefix-sharing workloads for reproducing the Feather paper claims.

Each request is a list of token ids. Prefix sharing is expressed *physically*: two
requests share a prefix iff they have the exact same leading token ids, which is what
vLLM's prefix cache keys on. That shared prefix therefore occupies a single set of KV
blocks in GPU memory that every request in the group reads from -- the locality effect
the paper attributes its bandwidth gains to.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Keep generated ids well clear of special/reserved tokens at either end of the vocab.
_VOCAB_MARGIN = 100


@dataclass
class Workload:
    """A batch of requests plus the distinct prefixes used (for cache warmup)."""

    name: str
    requests: list[list[int]]
    shared_prefixes: list[list[int]]
    description: str
    prefix_group_ids: list[int] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def avg_prompt_len(self) -> float:
        if not self.requests:
            return 0.0
        return sum(len(r) for r in self.requests) / len(self.requests)


def _rand_tokens(rng: random.Random, n: int, vocab_size: int) -> list[int]:
    lo, hi = _VOCAB_MARGIN, max(_VOCAB_MARGIN + 1, vocab_size - _VOCAB_MARGIN)
    return [rng.randrange(lo, hi) for _ in range(n)]


def _make_prefix(group: int, length: int, vocab_size: int, seed: int) -> list[int]:
    """A random prefix whose first token uniquely encodes its group, so distinct
    groups never accidentally share leading tokens (which would merge in the cache)."""
    rng = random.Random(f"{seed}:prefix:{group}")
    tokens = _rand_tokens(rng, length, vocab_size)
    if tokens:
        tokens[0] = _VOCAB_MARGIN + (group % (vocab_size - 2 * _VOCAB_MARGIN))
    return tokens


def _make_suffix(uid: int, length: int, vocab_size: int, seed: int) -> list[int]:
    """A per-request-unique suffix, so every request is a distinct sequence even when
    it shares a prefix with others (its first token encodes the global request id)."""
    rng = random.Random(f"{seed}:suffix:{uid}")
    tokens = _rand_tokens(rng, length, vocab_size)
    if tokens:
        tokens[0] = _VOCAB_MARGIN + (uid % (vocab_size - 2 * _VOCAB_MARGIN))
    return tokens


def build_homogeneity_fraction(
    beta: float,
    num_requests: int,
    prefix_len: int,
    suffix_len: int,
    vocab_size: int,
    seed: int = 0,
) -> Workload:
    """Experiment 1 / Figure 4: two prefixes A and B, each `prefix_len` tokens.

    A fraction `beta` of requests use prefix A, the rest use prefix B; each request
    gets a unique suffix. beta in {0, 1} is fully homogeneous; otherwise the batch
    contains two coexisting prefix groups.
    """
    prefix_a = _make_prefix(0, prefix_len, vocab_size, seed)
    prefix_b = _make_prefix(1, prefix_len, vocab_size, seed)

    n_a = round(beta * num_requests)
    requests: list[list[int]] = []
    prefix_group_ids: list[int] = []
    for uid in range(num_requests):
        group = 0 if uid < n_a else 1
        prefix = prefix_a if group == 0 else prefix_b
        requests.append(prefix + _make_suffix(uid, suffix_len, vocab_size, seed))
        prefix_group_ids.append(group)

    return Workload(
        name="homogeneity_fraction",
        requests=requests,
        shared_prefixes=[prefix_a, prefix_b],
        description=f"beta={beta} ({n_a}/{num_requests} on prefix A), prefix_len={prefix_len}",
        prefix_group_ids=prefix_group_ids,
        params={"beta": beta, "n_a": n_a, "num_requests": num_requests, "prefix_len": prefix_len},
    )


def build_shared_length(
    share_fraction: float,
    num_requests: int,
    total_len: int,
    vocab_size: int,
    seed: int = 0,
) -> Workload:
    """Experiment 2 / Figure 5: every request is `total_len` tokens and shares a common
    prefix of length `share_fraction * total_len`; the remaining tokens are unique.
    """
    shared_len = int(round(share_fraction * total_len))
    shared_len = max(0, min(total_len, shared_len))
    shared_prefix = _make_prefix(0, shared_len, vocab_size, seed)

    requests: list[list[int]] = []
    prefix_group_ids: list[int] = []
    for uid in range(num_requests):
        unique = _make_suffix(uid, total_len - shared_len, vocab_size, seed)
        requests.append(shared_prefix + unique)
        prefix_group_ids.append(0 if shared_len else uid)

    return Workload(
        name="shared_length",
        requests=requests,
        shared_prefixes=[shared_prefix] if shared_len else [],
        description=f"share_fraction={share_fraction}, shared_len={shared_len}/{total_len}",
        prefix_group_ids=prefix_group_ids,
        params={
            "share_fraction": share_fraction,
            "shared_len": shared_len,
            "total_len": total_len,
            "num_requests": num_requests,
        },
    )


def build_num_groups(
    num_groups: int,
    num_requests: int,
    prefix_len: int,
    suffix_len: int,
    vocab_size: int,
    seed: int = 0,
) -> Workload:
    """Experiment 3 / Figure 6: `num_groups` distinct prefixes, each `prefix_len` tokens,
    with requests distributed equally across the groups.
    """
    prefixes = [_make_prefix(g, prefix_len, vocab_size, seed) for g in range(num_groups)]

    requests: list[list[int]] = []
    prefix_group_ids: list[int] = []
    for uid in range(num_requests):
        group = uid % num_groups
        prefix = prefixes[group]
        requests.append(prefix + _make_suffix(uid, suffix_len, vocab_size, seed))
        prefix_group_ids.append(group)

    return Workload(
        name="num_groups",
        requests=requests,
        shared_prefixes=prefixes,
        description=f"num_groups={num_groups}, prefix_len={prefix_len}, requests={num_requests}",
        prefix_group_ids=prefix_group_ids,
        params={"num_groups": num_groups, "prefix_len": prefix_len, "num_requests": num_requests},
    )
