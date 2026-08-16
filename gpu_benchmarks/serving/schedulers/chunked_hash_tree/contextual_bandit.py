"""Contextual-bandit batching policy from the supplied Feather vLLM fork."""

import math


class ContextualBanditScheduler:
    def __init__(self, exploration_factor: float = 200.0):
        self.c = exploration_factor
        self.action_counts: dict[tuple[str, bool], int] = {}
        self.action_rewards: dict[tuple[str, bool], float] = {}
        self.total_steps = 0

    @staticmethod
    def discretize_state(
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
    ) -> str:
        chunk_diff = chunks_before - chunks_after
        batch_bin = min(batch_size // 4, 10)
        chunk_diff_bin = max(-5, min(5, chunk_diff))
        requests_bin = min(requests_waiting // 4, 10)
        return f"{batch_bin}_{chunk_diff_bin}_{requests_bin}"

    def get_ucb_score(self, state: str, action: bool) -> float:
        key = (state, action)
        if key not in self.action_counts or self.action_counts[key] == 0:
            return float("inf")
        count = self.action_counts[key]
        average_reward = self.action_rewards[key] / count
        exploration_bonus = self.c * math.sqrt(math.log(self.total_steps + 1) / count)
        return average_reward + exploration_bonus

    def should_add_request(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
    ) -> bool:
        if batch_size == 0:
            return True
        if chunks_before == chunks_after and chunks_before > 0:
            return True

        state = self.discretize_state(
            batch_size,
            chunks_before,
            chunks_after,
            requests_waiting,
        )
        return self.get_ucb_score(state, True) >= self.get_ucb_score(state, False)

    def update(
        self,
        batch_size: int,
        chunks_before: int,
        chunks_after: int,
        requests_waiting: int,
        action: bool,
        reward: float,
    ) -> None:
        state = self.discretize_state(
            batch_size,
            chunks_before,
            chunks_after,
            requests_waiting,
        )
        key = (state, action)
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        self.action_rewards[key] = self.action_rewards.get(key, 0.0) + reward
        self.total_steps += 1
