# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Circle-packing DomainSpec for ESN."""

from __future__ import annotations

from typing import Any

import numpy as np

from esn import DomainSpec, EvaluationDiagnostics, EvaluationResult, UvSandboxCompiler

from .evaluator import N_CIRCLES, _validate_packing
from .initial import INITIAL_SOLUTION


def _initial_program_code() -> str:
    code = INITIAL_SOLUTION.strip()
    return code.replace("def construct_packing():", "def solve():", 1)


def _repair_packing(centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cheap, deterministic feasibility projection (no LLM, ~microseconds).

    Converts a near-miss packing (overlaps / out-of-bounds) into a *valid* one by
    clipping centers into the square, capping each radius at its wall distance,
    and shrinking overlapping pairs proportionally for a few passes. This is the
    seed program's own overlap-resolution logic, reused so the search gets a
    valid-but-suboptimal candidate to refine instead of discarding a near-miss."""
    centers = np.clip(np.asarray(centers, dtype=float), 0.0, 1.0)
    radii = np.maximum(np.asarray(radii, dtype=float), 0.0).copy()
    n = len(radii)
    for i in range(n):
        x, y = centers[i]
        radii[i] = max(0.0, min(radii[i], x, y, 1.0 - x, 1.0 - y))
    for _ in range(25):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.hypot(*(centers[i] - centers[j])))
                if radii[i] + radii[j] - dist > 1e-9:
                    scale = dist / (radii[i] + radii[j] + 1e-12)
                    radii[i] *= scale
                    radii[j] *= scale
                    moved = True
        if not moved:
            break
    return centers, np.maximum(radii, 0.0)


def evaluate_circle_packing_artifact_repaired(artifact: Any) -> EvaluationResult:
    """Like :func:`evaluate_circle_packing_artifact` but projects the candidate
    into the feasible region first (see :func:`_repair_packing`). Genuinely
    malformed returns (wrong arity/shape) still fail — repair fixes geometry,
    not broken plumbing."""
    diagnostics = EvaluationDiagnostics(
        constraints={"container": "unit_square", "n_circles": float(N_CIRCLES), "repaired": 1.0}
    )
    try:
        if not isinstance(artifact, (tuple, list)) or len(artifact) not in (2, 3):
            diagnostics.violations.append("solve() must return (centers, radii)[, score]")
            return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)
        centers = np.asarray(artifact[0], dtype=float)
        radii = np.asarray(artifact[1], dtype=float)
        if centers.shape != (N_CIRCLES, 2) or radii.shape != (N_CIRCLES,):
            diagnostics.violations.append(
                f"shape mismatch centers={centers.shape} radii={radii.shape}"
            )
            return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)
        centers, radii = _repair_packing(centers, radii)
        err = _validate_packing(centers, radii)
        if err:  # should not happen post-repair, but stay safe
            diagnostics.violations.append(f"post-repair invalid: {err}")
            return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)
        score = float(np.sum(radii))
        diagnostics.residuals["sum_of_radii"] = score
        return EvaluationResult(score=score, success=True, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics.violations.append(f"{type(exc).__name__}: {exc}")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)


def evaluate_circle_packing_artifact(artifact: Any) -> EvaluationResult:
    """Evaluate a materialized packing artifact returned by `solve()`."""
    diagnostics = EvaluationDiagnostics(
        constraints={"container": "unit_square", "n_circles": float(N_CIRCLES)}
    )

    try:
        if not isinstance(artifact, (tuple, list)) or len(artifact) not in (2, 3):
            diagnostics.violations.append(
                "solve() must return (centers, radii) or (centers, radii, score)"
            )
            return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

        centers = np.asarray(artifact[0], dtype=float)
        radii = np.asarray(artifact[1], dtype=float)
        err = _validate_packing(centers, radii)
        if err:
            diagnostics.violations.append(err)
            return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

        score = float(np.sum(radii))
        diagnostics.residuals["sum_of_radii"] = score
        return EvaluationResult(score=score, success=True, diagnostics=diagnostics)
    except Exception as exc:
        diagnostics.violations.append(f"{type(exc).__name__}: {exc}")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)


def create_circle_packing_domain_spec(
    *,
    timeout_seconds: int = 60,
    repair: bool = False,
) -> DomainSpec:
    evaluator = (
        evaluate_circle_packing_artifact_repaired if repair else evaluate_circle_packing_artifact
    )
    return DomainSpec(
        name="circle_packing",
        description="Pack 26 non-overlapping circles in a unit square to maximize the sum of radii.",
        initial_code=_initial_program_code(),
        compiler=UvSandboxCompiler(
            allowed_imports=frozenset(
                {
                    "numpy",
                    "math",
                    "scipy",
                    "itertools",
                    "collections",
                    "functools",
                    "random",
                    "heapq",
                    "bisect",
                }
            ),
            max_lines=None,
            timeout_seconds=timeout_seconds,
            seed=42,
        ),
        evaluator=evaluator,
        allowed_imports=None,
        max_code_lines=None,
        hard_constraints=[
            "All circles must lie entirely within the unit square [0,1]x[0,1].",
            "No two circles may overlap.",
            "solve() must return (centers, radii) or (centers, radii, score).",
            "There must be exactly 26 circles.",
            "Program must complete within 60 seconds. Use bounded loops, avoid grid searches >200 evaluations.",
        ],
        examples=[_initial_program_code()],
        hints=[
            "Start from valid constructive layouts before aggressive refinement.",
            "Respect geometry constraints first, then grow radii greedily.",
            "scipy.optimize (L-BFGS-B, SLSQP) is available for numerical optimization.",
            "scipy.spatial for Voronoi, Delaunay triangulation.",
            "scipy.linprog for linear programming.",
        ],
        preferred_solution_shape=(
            "Prefer direct constructive solutions: compute or encode circle "
            "centers/radii using explicit geometric formulas, known layouts, "
            "or small deterministic refinements. Do not make solve() depend "
            "primarily on long-running global search, simulated annealing, "
            "basin hopping, or repeated scipy.optimize loops. If an optimizer "
            "is used, it should be a bounded polish step on top of an "
            "explicit construction, not the main method."
        ),
    )
