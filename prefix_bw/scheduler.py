"""Prefix-local batch planners for synthetic vLLM workloads."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar

from prefix_bw.workload import Workload

SCHEDULES = (
    "single",
    "prefix-grouped",
    "prefix-adaptive",
    "prefix-hash-adaptive",
    "prefix-hash-auto",
    "offline-prefix-wave",
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class _PrefixHashGroup:
    """A discovered shared-prefix group in request-index space."""

    members: tuple[int, ...]
    shared_len: int
    saved_work: int


def _chunked(items: list[_T], size: int) -> list[list[_T]]:
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_request_batches(
    workload: Workload,
    schedule: str,
    max_num_seqs: int,
    decode_tokens: int,
    prefix_batch_size: int | None,
    min_prefix_batch_size: int,
    prefix_hash_chunk_size: int,
    min_shared_prefix_len: int,
    prefix_auto_min_fill: float,
    offline_prefix_min_gain: float,
    offline_prefix_min_fill: float,
    offline_prefix_max_waves: int,
) -> list[list[list[int]]]:
    """Plan one or more generate batches for the requested schedule."""
    if schedule == "single":
        return [workload.requests]

    if schedule == "offline-prefix-wave":
        return _build_offline_prefix_wave_batches(
            workload,
            max_num_seqs,
            decode_tokens,
            prefix_batch_size,
            min_prefix_batch_size,
            prefix_hash_chunk_size,
            min_shared_prefix_len,
            offline_prefix_min_gain,
            offline_prefix_min_fill,
            offline_prefix_max_waves,
        )

    if schedule in {"prefix-hash-adaptive", "prefix-hash-auto"}:
        hash_batches = _build_prefix_hash_batches(
            workload,
            max_num_seqs,
            decode_tokens,
            prefix_batch_size,
            min_prefix_batch_size,
            prefix_hash_chunk_size,
            min_shared_prefix_len,
        )
        if schedule == "prefix-hash-auto" and _should_use_single_for_hash_batches(
            workload,
            hash_batches,
            max_num_seqs,
            prefix_batch_size,
            prefix_auto_min_fill,
        ):
            return [workload.requests]
        return hash_batches

    if schedule in {"prefix-grouped", "prefix-adaptive"}:
        return _build_oracle_prefix_batches(
            workload,
            schedule,
            max_num_seqs,
            prefix_batch_size,
            min_prefix_batch_size,
        )

    raise ValueError(f"unknown schedule {schedule!r}")


def _build_oracle_prefix_batches(
    workload: Workload,
    schedule: str,
    max_num_seqs: int,
    prefix_batch_size: int | None,
    min_prefix_batch_size: int,
) -> list[list[list[int]]]:
    if len(workload.prefix_group_ids) != len(workload.requests):
        raise ValueError(f"{schedule} scheduling requires one prefix group id per request")
    if min_prefix_batch_size <= 0:
        raise ValueError(f"min prefix batch size must be positive, got {min_prefix_batch_size}")

    groups: dict[int, list[list[int]]] = defaultdict(list)
    group_order: list[int] = []
    for tokens, group_id in zip(workload.requests, workload.prefix_group_ids, strict=True):
        if group_id not in groups:
            group_order.append(group_id)
        groups[group_id].append(tokens)

    batch_size = max_num_seqs if prefix_batch_size is None else prefix_batch_size
    batches: list[list[list[int]]] = []
    residual: list[list[int]] = []
    has_large_group = False
    for group_id in group_order:
        group = groups[group_id]
        if schedule == "prefix-adaptive" and len(group) < min_prefix_batch_size:
            residual.extend(group)
            continue
        has_large_group = True
        batches.extend(_chunked(group, batch_size))
    if schedule == "prefix-adaptive" and not has_large_group:
        return [workload.requests]
    if residual:
        batches.extend(_chunked(residual, batch_size))
    return batches


def _should_use_single_for_hash_batches(
    workload: Workload,
    batches: list[list[list[int]]],
    max_num_seqs: int,
    prefix_batch_size: int | None,
    min_fill: float,
) -> bool:
    if len(batches) <= 1:
        return False
    if not 0 <= min_fill <= 1:
        raise ValueError(f"prefix auto min fill must be in [0, 1], got {min_fill}")

    batch_size = max_num_seqs if prefix_batch_size is None else prefix_batch_size
    capacity = len(batches) * batch_size
    fill_ratio = workload.num_requests / capacity if capacity else 0.0
    return fill_ratio < min_fill


def _build_prefix_hash_batches(
    workload: Workload,
    max_num_seqs: int,
    decode_tokens: int,
    prefix_batch_size: int | None,
    min_prefix_batch_size: int,
    prefix_hash_chunk_size: int,
    min_shared_prefix_len: int,
) -> list[list[list[int]]]:
    """Discover prefix-homogeneous groups from token IDs, without oracle labels.

    This is a lightweight Chunked Hash Tree analogue: each prompt contributes nodes
    for fixed-size token chunks. Deep nodes with many members represent long shared
    prefixes. We greedily pick disjoint high-benefit nodes, where benefit scales
    with the amount of repeated KV-cache traversal avoided during decode.
    """
    min_score = max(1, min_prefix_batch_size - 1) * max(1, min_shared_prefix_len) * max(1, decode_tokens)
    selected_groups = _select_prefix_hash_groups(
        workload,
        decode_tokens,
        min_prefix_batch_size,
        prefix_hash_chunk_size,
        min_shared_prefix_len,
        min_score,
    )

    if not selected_groups:
        return [workload.requests]

    return _groups_to_batches(workload, selected_groups, max_num_seqs, prefix_batch_size)


def _select_prefix_hash_groups(
    workload: Workload,
    decode_tokens: int,
    min_prefix_batch_size: int,
    prefix_hash_chunk_size: int,
    min_shared_prefix_len: int,
    min_saved_work: int = 0,
) -> list[_PrefixHashGroup]:
    if min_prefix_batch_size <= 0:
        raise ValueError(f"min prefix batch size must be positive, got {min_prefix_batch_size}")
    if prefix_hash_chunk_size <= 0:
        raise ValueError(f"prefix hash chunk size must be positive, got {prefix_hash_chunk_size}")
    if min_shared_prefix_len < 0:
        raise ValueError(f"min shared prefix len must be non-negative, got {min_shared_prefix_len}")

    node_by_key: dict[tuple[int, tuple[int, ...]], int] = {}
    node_members: dict[int, list[int]] = defaultdict(list)
    node_lens: dict[int, int] = {}
    next_node_id = 1

    for request_idx, tokens in enumerate(workload.requests):
        parent = 0
        chunk_limit = (len(tokens) // prefix_hash_chunk_size) * prefix_hash_chunk_size
        for end in range(prefix_hash_chunk_size, chunk_limit + 1, prefix_hash_chunk_size):
            chunk = tuple(tokens[end - prefix_hash_chunk_size : end])
            key = (parent, chunk)
            node_id = node_by_key.get(key)
            if node_id is None:
                node_id = next_node_id
                next_node_id += 1
                node_by_key[key] = node_id
                node_lens[node_id] = end
            node_members[node_id].append(request_idx)
            parent = node_id

    candidates = [
        node_id
        for node_id, members in node_members.items()
        if len(members) >= min_prefix_batch_size and node_lens[node_id] >= min_shared_prefix_len
    ]
    candidates.sort(
        key=lambda node_id: (
            (len(node_members[node_id]) - 1) * node_lens[node_id] * max(1, decode_tokens),
            node_lens[node_id],
            len(node_members[node_id]),
        ),
        reverse=True,
    )

    selected_groups: list[_PrefixHashGroup] = []
    used_requests: set[int] = set()
    for node_id in candidates:
        members = tuple(idx for idx in node_members[node_id] if idx not in used_requests)
        if len(members) < min_prefix_batch_size:
            continue
        saved_work = (len(members) - 1) * node_lens[node_id] * max(1, decode_tokens)
        if saved_work < min_saved_work:
            continue
        selected_groups.append(_PrefixHashGroup(members, node_lens[node_id], saved_work))
        used_requests.update(members)

    selected_groups.sort(key=lambda group: min(group.members))
    return selected_groups


def _groups_to_batches(
    workload: Workload,
    selected_groups: list[_PrefixHashGroup],
    max_num_seqs: int,
    prefix_batch_size: int | None,
) -> list[list[list[int]]]:
    used_requests = {idx for group in selected_groups for idx in group.members}
    residual = [idx for idx in range(len(workload.requests)) if idx not in used_requests]
    batch_size = max_num_seqs if prefix_batch_size is None else prefix_batch_size

    batches: list[list[list[int]]] = []
    for members in selected_groups:
        group_requests = [workload.requests[idx] for idx in members.members]
        batches.extend(_chunked(group_requests, batch_size))
    if residual:
        residual_requests = [workload.requests[idx] for idx in residual]
        batches.extend(_chunked(residual_requests, batch_size))
    return batches


def _build_offline_prefix_wave_batches(
    workload: Workload,
    max_num_seqs: int,
    decode_tokens: int,
    prefix_batch_size: int | None,
    min_prefix_batch_size: int,
    prefix_hash_chunk_size: int,
    min_shared_prefix_len: int,
    min_gain: float,
    min_fill: float,
    max_waves: int,
) -> list[list[list[int]]]:
    """Plan conservative prefix-local waves, falling back to a single vLLM batch."""
    if min_gain < 0:
        raise ValueError(f"offline prefix min gain must be non-negative, got {min_gain}")
    if not 0 <= min_fill <= 1:
        raise ValueError(f"offline prefix min fill must be in [0, 1], got {min_fill}")
    if max_waves < 0:
        raise ValueError(f"offline prefix max waves must be non-negative, got {max_waves}")

    selected_groups = _select_prefix_hash_groups(
        workload,
        decode_tokens,
        min_prefix_batch_size,
        prefix_hash_chunk_size,
        min_shared_prefix_len,
    )
    if not selected_groups:
        return [workload.requests]

    batches = _groups_to_batches(workload, selected_groups, max_num_seqs, prefix_batch_size)
    if len(batches) <= 1:
        return [workload.requests]
    if max_waves and len(batches) > max_waves:
        return [workload.requests]

    batch_size = max_num_seqs if prefix_batch_size is None else prefix_batch_size
    total_capacity = len(batches) * batch_size
    fill = workload.num_requests / total_capacity if total_capacity else 0.0
    if fill < min_fill:
        return [workload.requests]

    total_prompt_work = sum(len(tokens) for tokens in workload.requests) * max(1, decode_tokens)
    saved_work = sum(group.saved_work for group in selected_groups)
    extra_waves = max(0, len(batches) - 1)
    wave_overhead_work = extra_waves * batch_size * max(1, decode_tokens) * max(1, min_shared_prefix_len)
    net_gain = (saved_work - wave_overhead_work) / total_prompt_work if total_prompt_work > 0 else 0.0
    if net_gain < min_gain:
        return [workload.requests]

    return batches
