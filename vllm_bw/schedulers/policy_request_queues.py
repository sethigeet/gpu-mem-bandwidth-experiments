"""vLLM request-queue adapters for the vendored scheduling policies."""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from vllm.v1.core.sched.build.chunked_hash_tree import (
    ChunkedHashTree,
    ChunkedHashTree_RL,
)
from vllm.v1.core.sched.chunked_hash_tree_python import (
    ChunkedHashTree as PythonChunkedHashTree,
)
from vllm.v1.core.sched.contextual_bandit import ContextualBanditScheduler
from vllm.v1.core.sched.radix_cost import TokenRadixTree
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

_FUNCTION_STATS: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "total_ns": 0})


def log_cycles(function):
    name = function.__qualname__

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = function(*args, **kwargs)
        elapsed = time.perf_counter_ns() - start
        stats = _FUNCTION_STATS[name]
        stats["calls"] += 1
        stats["total_ns"] += elapsed
        if stats["calls"] % 10 == 0:
            average = stats["total_ns"] / stats["calls"]
            print(f"[PROFILE] {name} | calls={stats['calls']} | total={stats['total_ns']} ns | avg={average:.1f} ns")
        return result

    return wrapper


class _StringTreeRequestQueue(RequestQueue):
    def __init__(self, tree: Any, maximum_cost: int):
        self._radix = tree
        self._maximum_cost = maximum_cost
        self._pending_requests: dict[str, Request] = {}

    @log_cycles
    def add_request(self, request: Request) -> None:
        self._radix.insert(request.request_id, list(request.all_token_ids))
        self._pending_requests[request.request_id] = request

    @log_cycles
    def pop_request(self) -> Request:
        request_id, _ = self._radix.find_best_request()
        if request_id is None:
            raise IndexError("pop from empty scheduler queue")
        request = self._pending_requests.pop(request_id)
        self._radix.activate_request(request_id)
        return request

    def peek_request(self) -> Request:
        request_id, _ = self._radix.find_best_request()
        if request_id is None:
            raise IndexError("peek from empty scheduler queue")
        return self._pending_requests[request_id]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        if request.request_id in self._pending_requests:
            self._radix.remove(request.request_id)
            del self._pending_requests[request.request_id]

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        request_id, _ = self._radix.find_best_request()
        return request_id is not None

    def __len__(self) -> int:
        return len(self._pending_requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._pending_requests.values())

    @log_cycles
    def free_request(self, request: Request) -> None:
        self._radix.finish_request(request.request_id)

    def should_add_more_to_batch(self, **kwargs) -> bool:
        if kwargs.get("current_batch_size", 0) == 0:
            return True
        request_id, cost = self._radix.find_best_request()
        return request_id is not None and cost <= self._maximum_cost


class RadixCostRequestQueue(_StringTreeRequestQueue):
    def __init__(self):
        super().__init__(TokenRadixTree(), maximum_cost=50)


class PythonChunkedHashTreeRequestQueue(_StringTreeRequestQueue):
    def __init__(self):
        super().__init__(PythonChunkedHashTree(chunk_size=500), maximum_cost=1)


class CppChunkedHashTreeRequestQueue(RequestQueue):
    def __init__(self):
        self._radix = ChunkedHashTree(chunk_size=500)
        self._pending_requests: dict[str, Request] = {}
        self._id_map: dict[str, int] = {}
        self._id_to_request: dict[int, str] = {}
        self._next_id = 1

    def _intern_id(self, request_id: str) -> int:
        integer_id = self._id_map.get(request_id)
        if integer_id is None:
            integer_id = self._next_id
            self._next_id += 1
            self._id_map[request_id] = integer_id
            self._id_to_request[integer_id] = request_id
        return integer_id

    def _release_id(self, request_id: str) -> None:
        integer_id = self._id_map.pop(request_id, None)
        if integer_id is not None:
            self._id_to_request.pop(integer_id, None)

    @log_cycles
    def add_request(self, request: Request) -> None:
        integer_id = self._intern_id(request.request_id)
        self._radix.insert(integer_id, request.all_token_ids)
        self._pending_requests[request.request_id] = request

    @log_cycles
    def pop_request(self) -> Request:
        integer_id, _ = self._radix.find_best_request()
        if integer_id == 0:
            raise IndexError("pop from empty scheduler queue")
        request_id = self._id_to_request[integer_id]
        request = self._pending_requests.pop(request_id)
        self._radix.activate_request(integer_id)
        return request

    def peek_request(self) -> Request:
        integer_id, _ = self._radix.find_best_request()
        if integer_id == 0:
            raise IndexError("peek from empty scheduler queue")
        return self._pending_requests[self._id_to_request[integer_id]]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        if request.request_id not in self._pending_requests:
            return
        integer_id = self._id_map.get(request.request_id)
        if integer_id is not None:
            self._radix.remove(integer_id)
        del self._pending_requests[request.request_id]
        self._release_id(request.request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        integer_id, _ = self._radix.find_best_request()
        return integer_id != 0

    def __len__(self) -> int:
        return len(self._pending_requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._pending_requests.values())

    @log_cycles
    def free_request(self, request: Request) -> None:
        integer_id = self._id_map.get(request.request_id)
        if integer_id is not None:
            self._radix.finish_request(integer_id)
            self._release_id(request.request_id)

    def should_add_more_to_batch(self, **kwargs) -> bool:
        if kwargs.get("current_batch_size", 0) == 0:
            return True
        integer_id, cost = self._radix.find_best_request()
        return integer_id != 0 and cost <= 1


class ChunkedHashTreeBanditRequestQueue(CppChunkedHashTreeRequestQueue):
    def __init__(self):
        super().__init__()
        self._radix = ChunkedHashTree_RL(chunk_size=50)
        self.scheduler = ContextualBanditScheduler()
        self._current_batch_steps: list[dict[str, int | bool]] = []
        self._last_seen_batch_id: int | None = None
        self._last_final_batch_size = 0

    def _find_best(self) -> tuple[int, int, int, int]:
        return self._radix.find_best_request()

    @log_cycles
    def pop_request(self) -> Request:
        integer_id, _, _, _ = self._find_best()
        if integer_id == 0:
            raise IndexError("pop from empty scheduler queue")
        request_id = self._id_to_request[integer_id]
        request = self._pending_requests.pop(request_id)
        self._radix.activate_request(integer_id)
        return request

    def peek_request(self) -> Request:
        integer_id, _, _, _ = self._find_best()
        if integer_id == 0:
            raise IndexError("peek from empty scheduler queue")
        return self._pending_requests[self._id_to_request[integer_id]]

    def __bool__(self) -> bool:
        integer_id, _, _, _ = self._find_best()
        return integer_id != 0

    @log_cycles
    def should_add_more_to_batch(self, **kwargs) -> bool:
        current_batch_size = kwargs.get("current_batch_size", 0)
        last_batch_time = kwargs.get("last_batch_time")
        current_batch_id = kwargs.get("current_batch_id")

        batch_changed = (
            current_batch_id is not None
            and self._last_seen_batch_id is not None
            and current_batch_id != self._last_seen_batch_id
        )
        if batch_changed and last_batch_time is not None and last_batch_time > 0 and self._current_batch_steps:
            self._provide_feedback(
                self._current_batch_steps,
                self._last_final_batch_size,
                last_batch_time,
            )
            self._current_batch_steps = []

        if current_batch_id is not None:
            self._last_seen_batch_id = current_batch_id
        if current_batch_size == 0:
            self._current_batch_steps = []
            return True

        integer_id, chunks_before, chunks_after, requests_waiting = self._find_best()
        if integer_id == 0:
            return False

        decision = self.scheduler.should_add_request(
            current_batch_size,
            chunks_before,
            chunks_after,
            requests_waiting,
        )
        self._current_batch_steps.append(
            {
                "batch_size": current_batch_size,
                "chunks_before": chunks_before,
                "chunks_after": chunks_after,
                "requests_waiting": requests_waiting,
                "action": decision,
            }
        )
        self._last_final_batch_size = current_batch_size
        return decision

    def _provide_feedback(
        self,
        steps: list[dict[str, int | bool]],
        final_batch_size: int,
        batch_execution_time: float,
    ) -> None:
        reward = final_batch_size / batch_execution_time
        for step in steps:
            self.scheduler.update(
                int(step["batch_size"]),
                int(step["chunks_before"]),
                int(step["chunks_after"]),
                int(step["requests_waiting"]),
                bool(step["action"]),
                reward,
            )
