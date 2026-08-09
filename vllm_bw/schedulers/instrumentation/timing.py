"""Low-overhead cumulative timing and NVTX ranges for vLLM investigations."""

from __future__ import annotations

import atexit
import functools
import os
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

_ENABLED = os.environ.get("VLLM_BW_PROFILE_SCHEDULER") == "1"
_GPU_RANGES_ENABLED = os.environ.get("VLLM_BW_PROFILE_GPU") == "1"
_GPU_EVENTS_ENABLED = os.environ.get("VLLM_BW_PROFILE_GPU_EVENTS") == "1"
_INTERVAL = max(1, int(os.environ.get("VLLM_BW_PROFILE_INTERVAL", "100")))
_STATS: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "total_ns": 0, "min_ns": 2**63 - 1, "max_ns": 0})
_GPU_EVENT_PAIRS: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
_GPU_EVENT_STATS: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0.0, "total_ms": 0.0, "min_ms": float("inf"), "max_ms": 0.0}
)


def _record(name: str, elapsed: int) -> None:
    stats = _STATS[name]
    stats["calls"] += 1
    stats["total_ns"] += elapsed
    stats["min_ns"] = min(stats["min_ns"], elapsed)
    stats["max_ns"] = max(stats["max_ns"], elapsed)
    if stats["calls"] % _INTERVAL == 0:
        _print_stats(name, stats)


def _print_stats(name: str, stats: dict[str, int]) -> None:
    print(
        "[VLLM_BW_TIMING] "
        f"name={name} "
        f"calls={stats['calls']} "
        f"total_ns={stats['total_ns']} "
        f"avg_ns={stats['total_ns'] / stats['calls']:.1f} "
        f"min_ns={stats['min_ns']} "
        f"max_ns={stats['max_ns']}",
        flush=True,
    )


def _print_final_stats() -> None:
    for name, stats in _STATS.items():
        if stats["calls"]:
            _print_stats(name, stats)


if _ENABLED:
    atexit.register(_print_final_stats)


@contextmanager
def profile_timing_region(name: str):
    """Measure one exclusive, explicitly named region when timing is enabled."""

    if not _ENABLED:
        yield
        return
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        _record(name, time.perf_counter_ns() - start)


def profile_scheduler_function(function: Callable[..., Any]) -> Callable[..., Any]:
    if not _ENABLED:
        return function

    name = getattr(function, "__qualname__", type(function).__qualname__)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            _record(name, time.perf_counter_ns() - start)

    return wrapper


def profile_gpu_range(name: str):
    """Add an NVTX push/pop range without importing torch during startup."""

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        if not (_GPU_RANGES_ENABLED or _GPU_EVENTS_ENABLED):
            return function

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import torch

            if _GPU_RANGES_ENABLED:
                torch.cuda.nvtx.range_push(name)
            start_event = end_event = None
            if _GPU_EVENTS_ENABLED:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            try:
                return function(*args, **kwargs)
            finally:
                if start_event is not None and end_event is not None:
                    end_event.record()
                    pairs = _GPU_EVENT_PAIRS[name]
                    pairs.append((start_event, end_event))
                    if len(pairs) >= _INTERVAL:
                        end_event.synchronize()
                        elapsed_values = [start.elapsed_time(end) for start, end in pairs]
                        stats = _GPU_EVENT_STATS[name]
                        stats["calls"] += len(elapsed_values)
                        stats["total_ms"] += sum(elapsed_values)
                        stats["min_ms"] = min(stats["min_ms"], *elapsed_values)
                        stats["max_ms"] = max(stats["max_ms"], *elapsed_values)
                        pairs.clear()
                        print(
                            "[VLLM_BW_GPU_TIMING] "
                            f"name={name} calls={int(stats['calls'])} "
                            f"total_ms={stats['total_ms']:.6f} "
                            f"avg_ms={stats['total_ms'] / stats['calls']:.6f} "
                            f"min_ms={stats['min_ms']:.6f} "
                            f"max_ms={stats['max_ms']:.6f}",
                            flush=True,
                        )
                if _GPU_RANGES_ENABLED:
                    torch.cuda.nvtx.range_pop()

        return wrapper

    return decorate
