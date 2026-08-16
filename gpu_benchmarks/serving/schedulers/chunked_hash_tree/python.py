"""Python Chunked Hash Tree from the supplied Feather vLLM fork.

Source: ``vllm/v1/core/sched/radix_tree.py`` at policy commit
``44e9e2ee955bba1c669777eff5f01d20b48a0b85``.
"""

import heapq
from collections import defaultdict

import xxhash


class ChunkedHashTree:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.levels: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.request_tokens: dict[str, list[int]] = {}
        self.request_hashes: dict[str, list[int]] = {}
        self.working_set_hashes: set[tuple[int, int]] = set()
        self.hash_ref_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.active_requests: set[str] = set()
        self.waiting_requests: list[str] = []
        self.missing_chunks_cache: dict[str, int] = {}
        self.hash_to_waiting_requests: dict[tuple[int, int], set[str]] = defaultdict(set)
        self.waiting_heap: list[tuple[int | float, str]] = []
        self.heap_entry_finder: dict[str, tuple[int | float, str]] = {}
        self.REMOVED = "<removed>"

    def _compute_hashes_incremental(self, token_ids: list[int]) -> list[int]:
        hashes = []
        hasher = xxhash.xxh64()
        for index, token in enumerate(token_ids):
            hasher.update(token.to_bytes(4, "little"))
            if (index + 1) % self.chunk_size == 0 or index == len(token_ids) - 1:
                hashes.append(hasher.intdigest())
        return hashes

    def _add_to_heap(self, request_id: str, missing_count: int | float) -> None:
        if request_id in self.heap_entry_finder:
            old_entry = self.heap_entry_finder[request_id]
            old_entry = (old_entry[0], self.REMOVED)
        entry = (missing_count, request_id)
        self.heap_entry_finder[request_id] = entry
        heapq.heappush(self.waiting_heap, entry)

    def _remove_from_heap(self, request_id: str) -> None:
        if request_id in self.heap_entry_finder:
            del self.heap_entry_finder[request_id]

    def insert(self, request_id: str, token_ids: list[int]) -> int:
        if request_id in self.request_tokens:
            return 0
        hashes = self._compute_hashes_incremental(token_ids)
        self.request_tokens[request_id] = token_ids
        self.request_hashes[request_id] = hashes
        for level, hash_value in enumerate(hashes):
            self.levels[level][hash_value].add(request_id)
        self.waiting_requests.append(request_id)
        missing = sum(
            1 for level, hash_value in enumerate(hashes) if (level, hash_value) not in self.working_set_hashes
        )
        self.missing_chunks_cache[request_id] = missing
        for level, hash_value in enumerate(hashes):
            self.hash_to_waiting_requests[(level, hash_value)].add(request_id)
        self._add_to_heap(request_id, missing)
        return len(token_ids)

    def find_best_request(self) -> tuple[str | None, int]:
        while self.waiting_heap:
            cost, request_id = self.waiting_heap[0]
            if request_id == self.REMOVED:
                heapq.heappop(self.waiting_heap)
                continue
            if request_id not in self.heap_entry_finder:
                heapq.heappop(self.waiting_heap)
                continue
            current_cost = self.missing_chunks_cache.get(request_id, float("inf"))
            if cost != current_cost:
                heapq.heappop(self.waiting_heap)
                self._add_to_heap(request_id, current_cost)
                continue
            return request_id, int(cost)
        return None, 0

    def activate_request(self, request_id: str) -> tuple[int, int]:
        if request_id in self.active_requests:
            return 0, 0
        if request_id not in self.request_hashes:
            return 0, 0
        self.active_requests.add(request_id)
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
            self.missing_chunks_cache.pop(request_id, None)
            self._remove_from_heap(request_id)

        hashes = self.request_hashes[request_id]
        added_count = 0
        for level, hash_value in enumerate(hashes):
            key = (level, hash_value)
            if self.hash_ref_counts[key] == 0:
                for waiting_request in self.hash_to_waiting_requests[key]:
                    if waiting_request in self.missing_chunks_cache:
                        old_missing = self.missing_chunks_cache[waiting_request]
                        self.missing_chunks_cache[waiting_request] -= 1
                        self._add_to_heap(waiting_request, old_missing - 1)
                self.working_set_hashes.add(key)
                added_count += 1
            self.hash_ref_counts[key] += 1
        for level, hash_value in enumerate(hashes):
            self.hash_to_waiting_requests[(level, hash_value)].discard(request_id)
        return added_count, len(hashes)

    def finish_request(self, request_id: str) -> tuple[int, int]:
        if request_id not in self.active_requests:
            return 0, 0
        self.active_requests.remove(request_id)
        hashes = self.request_hashes.get(request_id, [])
        evicted_count = 0
        for level, hash_value in enumerate(hashes):
            key = (level, hash_value)
            self.hash_ref_counts[key] -= 1
            if self.hash_ref_counts[key] == 0:
                for waiting_request in self.hash_to_waiting_requests[key]:
                    if waiting_request in self.missing_chunks_cache:
                        old_missing = self.missing_chunks_cache[waiting_request]
                        self.missing_chunks_cache[waiting_request] += 1
                        self._add_to_heap(waiting_request, old_missing + 1)
                self.working_set_hashes.discard(key)
                evicted_count += 1
                del self.hash_ref_counts[key]
        self.remove(request_id)
        return evicted_count, len(hashes)

    def remove(self, request_id: str) -> None:
        if request_id not in self.request_tokens:
            return
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
            self._remove_from_heap(request_id)
        self.missing_chunks_cache.pop(request_id, None)
        hashes = self.request_hashes.get(request_id, [])
        for level, hash_value in enumerate(hashes):
            key = (level, hash_value)
            self.hash_to_waiting_requests[key].discard(request_id)
            if not self.hash_to_waiting_requests[key]:
                del self.hash_to_waiting_requests[key]
        for level, hash_value in enumerate(hashes):
            self.levels[level][hash_value].discard(request_id)
            if not self.levels[level][hash_value]:
                del self.levels[level][hash_value]
        del self.request_tokens[request_id]
        del self.request_hashes[request_id]

    def get_working_set_size(self) -> int:
        return len(self.working_set_hashes)

    def get_working_set_tokens(self) -> int:
        return len(self.working_set_hashes) * self.chunk_size

    def get_stats(self) -> dict[str, int | float]:
        total_chunks = sum(len(hashes) for hashes in self.request_hashes.values())
        request_count = len(self.request_tokens)
        return {
            "total_requests": request_count,
            "active_requests": len(self.active_requests),
            "waiting_requests": len(self.waiting_requests),
            "working_set_chunks": len(self.working_set_hashes),
            "working_set_tokens_estimate": self.get_working_set_tokens(),
            "total_chunks": total_chunks,
            "avg_chunks_per_request": (total_chunks / request_count if request_count else 0),
        }
