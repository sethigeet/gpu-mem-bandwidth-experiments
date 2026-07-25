"""Token-level radix-cost tree from the supplied Feather vLLM fork."""

from __future__ import annotations

from collections import defaultdict, deque


class _RadixNode:
    __slots__ = (
        "tokens",
        "parent",
        "children",
        "requests",
        "ref_count",
        "in_working_set",
    )

    def __init__(self, tokens: list[int], parent: _RadixNode | None = None):
        self.tokens = tokens
        self.parent = parent
        self.children: dict[int, _RadixNode] = {}
        self.requests: set[str] = set()
        self.ref_count = 0
        self.in_working_set = False


class TokenRadixTree:
    def __init__(self):
        self.root = _RadixNode(tokens=[])
        self.root.in_working_set = True
        self.request_to_nodes: dict[str, list[_RadixNode]] = {}
        self.active_requests: set[str] = set()
        self.working_set_nodes: set[_RadixNode] = {self.root}
        self.working_set_ref_count: dict[_RadixNode, int] = defaultdict(int)
        self.waiting_requests: list[str] = []

    @staticmethod
    def _find_common_prefix_length(tokens1: list[int], tokens2: list[int]) -> int:
        for index in range(min(len(tokens1), len(tokens2))):
            if tokens1[index] != tokens2[index]:
                return index
        return min(len(tokens1), len(tokens2))

    def insert(self, request_id: str, token_ids: list[int]) -> int:
        if request_id in self.request_to_nodes:
            raise ValueError(f"Request {request_id} already exists")
        node = self.root
        remaining = token_ids
        nodes_traversed = [self.root]

        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                new_node = _RadixNode(tokens=list(remaining), parent=node)
                node.children[first_token] = new_node
                new_node.requests.add(request_id)
                new_node.ref_count += 1
                nodes_traversed.append(new_node)
                break

            child = node.children[first_token]
            common_length = self._find_common_prefix_length(remaining, child.tokens)
            if common_length == len(child.tokens):
                child.requests.add(request_id)
                child.ref_count += 1
                nodes_traversed.append(child)
                remaining = remaining[common_length:]
                node = child
                continue

            common_node = _RadixNode(
                tokens=child.tokens[:common_length],
                parent=node,
            )
            common_node.requests = child.requests.copy()
            common_node.ref_count = child.ref_count
            child.tokens = child.tokens[common_length:]
            child.parent = common_node
            if child.tokens:
                common_node.children[child.tokens[0]] = child

            common_node.in_working_set = child.in_working_set
            if common_node.in_working_set:
                self.working_set_nodes.add(common_node)
                if child in self.working_set_ref_count:
                    self.working_set_ref_count[common_node] = self.working_set_ref_count[child]
            node.children[first_token] = common_node
            common_node.requests.add(request_id)
            common_node.ref_count += 1
            nodes_traversed.append(common_node)

            for existing_request_id in child.requests:
                if existing_request_id in self.request_to_nodes and existing_request_id != request_id:
                    request_nodes = self.request_to_nodes[existing_request_id]
                    new_path = []
                    for request_node in request_nodes:
                        if request_node is child:
                            new_path.append(common_node)
                        new_path.append(request_node)
                    self.request_to_nodes[existing_request_id] = new_path

            remaining = remaining[common_length:]
            if remaining:
                new_node = _RadixNode(tokens=list(remaining), parent=common_node)
                common_node.children[remaining[0]] = new_node
                new_node.requests.add(request_id)
                new_node.ref_count += 1
                nodes_traversed.append(new_node)
            break

        self.request_to_nodes[request_id] = nodes_traversed
        self.waiting_requests.append(request_id)
        return len(token_ids)

    def _count_missing_tokens(self, request_id: str) -> int:
        return sum(
            len(node.tokens) for node in self.request_to_nodes.get(request_id, []) if node not in self.working_set_nodes
        )

    def find_best_request(self) -> tuple[str | None, float]:
        if not self.waiting_requests:
            return None, 0
        best_request = None
        best_cost = float("inf")
        queue = deque()
        visited = set()
        for node in self.working_set_nodes:
            is_leaf = not any(child in self.working_set_nodes for child in node.children.values())
            if is_leaf:
                queue.append(node)

        while queue and best_cost > 0:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for request_id in node.requests:
                if request_id not in self.waiting_requests or request_id in self.active_requests:
                    continue
                cost = self._count_missing_tokens(request_id)
                if cost < best_cost:
                    best_cost = cost
                    best_request = request_id
                    if cost == 0:
                        return best_request, best_cost
            if node.parent and node.parent not in visited:
                queue.append(node.parent)

        if best_request is None:
            best_request = self.waiting_requests[0]
            return best_request, self._count_missing_tokens(best_request)
        return best_request, best_cost

    def activate_request(self, request_id: str) -> tuple[int, int]:
        if request_id in self.active_requests:
            return 0, 0
        self.active_requests.add(request_id)
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
        nodes = self.request_to_nodes[request_id]
        added_count = 0
        for node in nodes:
            if self.working_set_ref_count[node] == 0:
                node.in_working_set = True
                self.working_set_nodes.add(node)
                added_count += 1
            self.working_set_ref_count[node] += 1
        return added_count, len(nodes)

    def finish_request(self, request_id: str) -> tuple[int, int]:
        if request_id not in self.active_requests:
            return 0, 0
        self.active_requests.remove(request_id)
        nodes = self.request_to_nodes[request_id]
        evicted_count = 0
        for node in nodes:
            self.working_set_ref_count[node] -= 1
            if self.working_set_ref_count[node] == 0:
                node.in_working_set = False
                self.working_set_nodes.discard(node)
                evicted_count += 1
        self.remove(request_id)
        return evicted_count, len(nodes)

    def remove(self, request_id: str) -> None:
        nodes = self.request_to_nodes.pop(request_id, None)
        if not nodes:
            return
        if request_id in self.waiting_requests:
            self.waiting_requests.remove(request_id)
        for node in reversed(nodes):
            if node is self.root:
                continue
            node.requests.discard(request_id)
            node.ref_count -= 1
            if node.ref_count == 0 and node.parent:
                if node.tokens and node.tokens[0] in node.parent.children:
                    del node.parent.children[node.tokens[0]]
                if node.in_working_set:
                    self.working_set_nodes.discard(node)
            else:
                break
