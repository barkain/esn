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
) -> DomainSpec:
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
        evaluator=evaluate_circle_packing_artifact,
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
