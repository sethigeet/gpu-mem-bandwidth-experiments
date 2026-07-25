"""Low-overhead cumulative timing for scheduler functions."""

from __future__ import annotations

import functools
import os
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

_ENABLED = os.environ.get("VLLM_BW_PROFILE_SCHEDULER") == "1"
_INTERVAL = max(1, int(os.environ.get("VLLM_BW_PROFILE_INTERVAL", "100")))
_STATS: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "total_ns": 0, "min_ns": 2**63 - 1, "max_ns": 0})


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
            elapsed = time.perf_counter_ns() - start
            stats = _STATS[name]
            stats["calls"] += 1
            stats["total_ns"] += elapsed
            stats["min_ns"] = min(stats["min_ns"], elapsed)
            stats["max_ns"] = max(stats["max_ns"], elapsed)
            if stats["calls"] % _INTERVAL == 0:
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

    return wrapper
