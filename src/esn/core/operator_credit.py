# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Operator credit model with UCB1-style sampling for ESN core."""

from __future__ import annotations

import math
import random
from typing import Any

from esn.core.enums import SearchMode
from esn.core.models import OperatorStats


class OperatorCreditModel:
    """Tracks per-operator effectiveness and provides UCB1-style sampling."""

    def __init__(self) -> None:
        self._stats: dict[str, OperatorStats] = {}

    def record(
        self,
        operator_name: str,
        compile_success: bool,
        eval_success: bool,
        score_delta: float = 0.0,
        epistemic_novelty: float = 0.0,
        spectral_novelty: float = 0.0,
        generation: int = 0,
    ) -> None:
        stats = self._stats.get(operator_name, OperatorStats())
        n = stats.attempts
        alpha = 0.3
        recent_score_delta = (
            score_delta if n == 0 else (1 - alpha) * stats.recent_score_delta + alpha * score_delta
        )
        non_improving_streak = (
            stats.non_improving_streak + 1 if eval_success and score_delta <= 1e-9 else 0
        )
        self._stats[operator_name] = OperatorStats(
            attempts=n + 1,
            compile_successes=stats.compile_successes + int(compile_success),
            eval_successes=stats.eval_successes + int(eval_success),
            mean_score_delta=(stats.mean_score_delta * n + score_delta) / (n + 1),
            recent_score_delta=recent_score_delta,
            mean_epistemic_novelty=(stats.mean_epistemic_novelty * n + epistemic_novelty) / (n + 1),
            mean_spectral_novelty=(stats.mean_spectral_novelty * n + spectral_novelty) / (n + 1),
            non_improving_streak=non_improving_streak,
            last_used_generation=max(stats.last_used_generation, generation),
        )

    def get_stats(self, operator_name: str) -> OperatorStats:
        return self._stats.get(operator_name, OperatorStats())

    def get_all_stats(self) -> dict[str, OperatorStats]:
        return dict(self._stats)

    def sample_operator(
        self, eligible: list[str], mode: SearchMode, exploration_rate: float = 0.15
    ) -> str:
        if not eligible:
            raise ValueError("No eligible operators")

        # Epsilon-greedy exploration floor: with probability exploration_rate,
        # pick a random operator regardless of UCB scores
        if random.random() < exploration_rate:  # noqa: S311
            return random.choice(eligible)  # noqa: S311

        total_attempts = sum(self._stats.get(n, OperatorStats()).attempts for n in eligible)
        if total_attempts == 0:
            return random.choice(eligible)

        c = 1.41  # exploration constant

        def ucb_score(name: str) -> float:
            stats = self._stats.get(name, OperatorStats())
            if stats.attempts == 0:
                return float("inf")
            if mode == SearchMode.EXPLOIT:
                reward = 0.5 * stats.mean_score_delta + 0.5 * stats.recent_score_delta
            elif mode == SearchMode.EXPLORE:
                reward = 0.7 * stats.mean_epistemic_novelty + 0.3 * stats.recent_score_delta
            elif mode == SearchMode.REPAIR:
                reward = stats.eval_successes / stats.attempts
            else:
                reward = 0.5 * stats.mean_score_delta + 0.5 * stats.recent_score_delta
            exploration = c * math.sqrt(math.log(total_attempts) / stats.attempts)
            streak_penalty = min(0.5, 0.05 * stats.non_improving_streak)
            return reward + exploration - streak_penalty

        return max(eligible, key=ucb_score)

    def to_dict(self) -> dict[str, Any]:
        return {name: stats.model_dump() for name, stats in self._stats.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorCreditModel:
        model = cls()
        for name, stats_data in data.items():
            model._stats[name] = OperatorStats.model_validate(stats_data)
        return model
