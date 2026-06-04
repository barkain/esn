# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Deterministic instance bundle for an OpenEvolve-inspired TSP benchmark.

This is a lightweight ESN-native benchmark, not a verbatim port of OpenEvolve's
UTSP/C++ setup. Instances are generated deterministically from seeds and cached
on disk, then evaluated via mean tour length across train / validation splits.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_TRAIN_SEEDS = list(range(1000, 1008))
_DEFAULT_VAL_SEEDS = list(range(2000, 2004))


@dataclass
class InstanceBundle:
    """Manages cached TSP instances and train/validation splits."""

    train_seeds: list[int]
    val_seeds: list[int]
    instances_dir: Path
    num_cities: int = 120

    @classmethod
    def default(cls, base_dir: Path | None = None) -> InstanceBundle:
        if base_dir is None:
            base_dir = Path(__file__).parent
        return cls(
            train_seeds=list(_DEFAULT_TRAIN_SEEDS),
            val_seeds=list(_DEFAULT_VAL_SEEDS),
            instances_dir=base_dir / "instances",
            num_cities=120,
        )

    def _instance_path(self, seed: int) -> Path:
        return self.instances_dir / f"seed_{seed}.txt"

    def _generate_seed(self, seed: int) -> None:
        rng = random.Random(seed)
        self.instances_dir.mkdir(parents=True, exist_ok=True)
        lines = [str(self.num_cities)]
        for _ in range(self.num_cities):
            x = rng.random()
            y = rng.random()
            lines.append(f"{x:.8f} {y:.8f}")
        self._instance_path(seed).write_text("\n".join(lines) + "\n")

    def ensure_generated(self) -> None:
        for seed in self.train_seeds + self.val_seeds:
            if not self._instance_path(seed).exists():
                self._generate_seed(seed)

    def get_input(self, seed: int) -> str:
        path = self._instance_path(seed)
        if not path.exists():
            self._generate_seed(seed)
        return path.read_text()

    @staticmethod
    def _parse_points(input_text: str) -> list[tuple[float, float]]:
        tokens = input_text.split()
        if not tokens:
            raise ValueError("empty instance")
        n = int(tokens[0])
        coords = tokens[1:]
        if len(coords) != 2 * n:
            raise ValueError(f"expected {2 * n} coordinates, got {len(coords)}")
        pts: list[tuple[float, float]] = []
        for i in range(n):
            pts.append((float(coords[2 * i]), float(coords[2 * i + 1])))
        return pts

    @staticmethod
    def _parse_tour(output: str, n: int) -> list[int]:
        toks = output.split()
        if len(toks) != n:
            raise ValueError(f"expected {n} tour indices, got {len(toks)}")
        try:
            tour = [int(tok) for tok in toks]
        except ValueError as exc:
            raise ValueError(f"non-integer tour token: {exc}") from exc
        if sorted(tour) != list(range(n)):
            raise ValueError("tour must be a permutation of 0..N-1")
        return tour

    @staticmethod
    def _tour_length(points: list[tuple[float, float]], tour: list[int]) -> float:
        total = 0.0
        n = len(tour)
        for i in range(n):
            x1, y1 = points[tour[i]]
            x2, y2 = points[tour[(i + 1) % n]]
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def score_output(self, seed: int, output: str) -> tuple[float, float, str]:
        """Return (score, length, error). Higher score is better."""
        input_text = self.get_input(seed)
        points = self._parse_points(input_text)
        n = len(points)
        try:
            tour = self._parse_tour(output, n)
            length = self._tour_length(points, tour)
            if not math.isfinite(length) or length <= 0.0:
                return 0.0, float("inf"), "invalid tour length"
            score = 1_000_000.0 / length
            return score, length, ""
        except Exception as exc:
            return 0.0, float("inf"), str(exc)

    def _evaluate_seeds(
        self,
        seeds: list[int],
        run_candidate: Callable[[str], str],
    ) -> dict:
        scores: dict[int, float] = {}
        lengths: dict[int, float] = {}
        errors: dict[int, str] = {}
        for seed in seeds:
            inp = self.get_input(seed)
            out = run_candidate(inp)
            score, length, err = self.score_output(seed, out)
            scores[seed] = score
            lengths[seed] = length
            if err:
                errors[seed] = err

        score_values = list(scores.values())
        finite_lengths = [v for v in lengths.values() if math.isfinite(v)]
        return {
            "scores": scores,
            "lengths": lengths,
            "errors": errors,
            "mean": sum(score_values) / len(score_values) if score_values else 0.0,
            "mean_length": sum(finite_lengths) / len(finite_lengths)
            if finite_lengths
            else float("inf"),
            "min": min(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
        }

    def evaluate_train(self, run_candidate: Callable[[str], str]) -> dict:
        return self._evaluate_seeds(self.train_seeds, run_candidate)

    def evaluate_val(self, run_candidate: Callable[[str], str]) -> dict:
        return self._evaluate_seeds(self.val_seeds, run_candidate)
