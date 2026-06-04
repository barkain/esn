# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""TSP DomainSpec for ESN.

A small Python-native Euclidean TSP variant with deterministic train/val
instance bundles and stdin/stdout candidate programs.
"""

from __future__ import annotations

from typing import Any

from esn import (
    CompilerResult,
    DomainSpec,
    EvaluationDiagnostics,
    EvaluationResult,
    StdioCompiler,
)

from .instance_bundle import InstanceBundle


class TSPCompiler:
    """Wraps StdioCompiler to score one candidate across all train seeds."""

    def __init__(self, stdio_compiler: StdioCompiler, bundle: InstanceBundle) -> None:
        self._compiler = stdio_compiler
        self._bundle = bundle

    def compile(self, code: str, seed: int = 42) -> CompilerResult:
        scores: dict[int, float] = {}
        lengths: dict[int, float] = {}
        errors: list[str] = []

        for train_seed in self._bundle.train_seeds:
            input_data = self._bundle.get_input(train_seed)
            result = self._compiler.compile(code, stdin_data=input_data, seed=seed)
            if not result.success:
                errors.append(f"Seed {train_seed}: {'; '.join(result.errors)}")
                scores[train_seed] = 0.0
                lengths[train_seed] = float("inf")
                continue

            score, length, score_err = self._bundle.score_output(train_seed, result.artifact)
            scores[train_seed] = score
            lengths[train_seed] = length
            if score_err:
                errors.append(f"Seed {train_seed} scoring: {score_err}")

        n_seeds = len(self._bundle.train_seeds)
        n_ok = sum(1 for s in scores.values() if s > 0)
        finite_lengths = [v for v in lengths.values() if v != float("inf")]
        mean_score = sum(scores.values()) / n_seeds if n_seeds else 0.0
        mean_length = sum(finite_lengths) / len(finite_lengths) if finite_lengths else float("inf")

        return CompilerResult(
            artifact={
                "scores": scores,
                "lengths": lengths,
                "mean": mean_score,
                "mean_length": mean_length,
            },
            success=n_ok > 0,
            errors=errors,
            metadata={
                "stage": "complete",
                "seeds_ok": n_ok,
                "seeds_total": n_seeds,
                "seed_scores": scores,
                "seed_lengths": lengths,
            },
        )


def evaluate_tsp_artifact(artifact: Any) -> EvaluationResult:
    diagnostics = EvaluationDiagnostics(
        constraints={"problem": "euclidean_tsp", "higher_is_better": 1.0}
    )
    if artifact is None or not isinstance(artifact, dict):
        diagnostics.violations.append("No valid artifact from compilation")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    mean_score = float(artifact.get("mean", 0.0))
    mean_length = float(artifact.get("mean_length", float("inf")))
    diagnostics.residuals["mean_score"] = mean_score
    diagnostics.residuals["mean_tour_length"] = mean_length
    lengths = artifact.get("lengths", {})
    if lengths:
        finite = [float(v) for v in lengths.values() if float(v) != float("inf")]
        if finite:
            diagnostics.residuals["min_length"] = min(finite)
            diagnostics.residuals["max_length"] = max(finite)
    return EvaluationResult(
        score=mean_score,
        success=mean_score > 0.0,
        diagnostics=diagnostics,
    )


def _initial_program_code() -> str:
    return """\
import math
import sys


def dist(points, i, j):
    x1, y1 = points[i]
    x2, y2 = points[j]
    return math.hypot(x2 - x1, y2 - y1)


def nearest_neighbor(points, start):
    n = len(points)
    unused = set(range(n))
    unused.remove(start)
    tour = [start]
    while unused:
        last = tour[-1]
        nxt = min(unused, key=lambda j: dist(points, last, j))
        unused.remove(nxt)
        tour.append(nxt)
    return tour


def tour_length(points, tour):
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist(points, tour[i], tour[(i + 1) % n])
    return total


def solve_instance(points):
    n = len(points)
    starts = [0, n // 3, (2 * n) // 3]
    best = None
    best_len = float("inf")
    for s in starts:
        tour = nearest_neighbor(points, s)
        length = tour_length(points, tour)
        if length < best_len:
            best = tour
            best_len = length
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    pts = []
    idx = 1
    for _ in range(n):
        pts.append((float(data[idx]), float(data[idx + 1])))
        idx += 2
    ans = solve_instance(pts)
    print(" ".join(str(x) for x in ans))


if __name__ == "__main__":
    main()
"""


def create_tsp_domain_spec(
    *,
    bundle: InstanceBundle | None = None,
    timeout_seconds: int = 20,
) -> DomainSpec:
    if bundle is None:
        bundle = InstanceBundle.default()

    compiler = TSPCompiler(
        stdio_compiler=StdioCompiler(
            timeout_seconds=timeout_seconds,
            max_lines=None,
            python_version="3.12",
        ),
        bundle=bundle,
    )

    return DomainSpec(
        name="tsp_tour_minimization",
        description=(
            "Euclidean TSP on fixed benchmark instances. Candidate programs read one "
            "instance from stdin and output a permutation representing a Hamiltonian cycle. "
            "Score is inverse mean tour length across train instances."
        ),
        initial_code=_initial_program_code(),
        compiler=compiler,
        evaluator=evaluate_tsp_artifact,
        program_interface="stdio",
        allowed_imports=None,
        max_code_lines=None,
        hard_constraints=[
            "The program reads one TSP instance from stdin and writes exactly one tour to stdout.",
            "Input format: first line N, then N lines of 'x y' floating-point coordinates in [0,1].",
            "Output format: exactly N integers separated by spaces, a permutation of 0..N-1.",
            "Every city must appear exactly once; do not repeat or omit nodes.",
            "The tour is implicitly closed: the evaluator adds the edge from last city back to first.",
            "Program must complete within 20 seconds per instance.",
            "Use only Python standard library.",
        ],
        examples=[_initial_program_code()],
        hints=[
            "Nearest-neighbor is a valid baseline but plateaus quickly.",
            "Look at insertion heuristics, candidate lists, 2-opt, restarts, and spatial clustering.",
            "The OpenEvolve example improves by reallocating compute between restart diversity and local refinement.",
            "Runtime matters: bounded local search is better than unbounded exact optimization.",
        ],
    )
