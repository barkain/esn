"""Bias-free circle_packing, NON-ZERO variant: every circle MUST have a strictly
positive radius. This kills the degenerate '25-circle r=0.1 grid + 1 wasted r=0
circle = 2.5' trick that made the 2.5 score a freebie. Prompt states the rule
AND the evaluator FAILS any packing with a circle of radius <= NZ_MIN_RADIUS.

NZ_MIN_RADIUS env (default 0.0 -> strict r>0). Raise it (e.g. 0.05) to also
break the 'grid + tiny interstitial filler' near-freebie, forcing real
non-uniform packing of all 26 circles.
"""
import os
from typing import Any

import numpy as np

from esn import DomainSpec, EvaluationDiagnostics, EvaluationResult, UvSandboxCompiler
from circle_packing.domain import evaluate_circle_packing_artifact, _initial_program_code

MIN_RADIUS = float(os.environ.get("NZ_MIN_RADIUS", "0.0"))
TIMEOUT = int(os.environ.get("NZ_TIMEOUT", "60"))
_seed_path = os.environ.get("NZ_SEED")
SEED_CODE = open(_seed_path).read() if _seed_path else _initial_program_code()

if MIN_RADIUS > 0:
    _rule = f"Every one of the 26 circles must have radius >= {MIN_RADIUS} (no degenerate or near-zero circles)."
else:
    _rule = "Every one of the 26 circles must have a strictly positive radius (radius > 0); a zero-radius circle is INVALID."

BARE_OBJECTIVE = (
    "Write a Python function construct_packing() that places 26 non-overlapping "
    "circles in the unit square [0,1] x [0,1] so as to maximize the sum of their "
    "radii. Return (centers, radii) where centers is a numpy array of shape "
    "(26, 2) of circle centers and radii is a numpy array of shape (26,) of radii. "
    + _rule
)
CONTRACT = [
    "Exactly 26 circles.",
    _rule,
    "No two circles may overlap.",
    "Every circle lies entirely within [0,1] x [0,1].",
    "Return (centers, radii) as numpy arrays of shape (26,2) and (26,).",
    f"The program must finish within {TIMEOUT} seconds.",
]


def evaluate_nonzero(artifact: Any) -> EvaluationResult:
    """Score via the base evaluator, but FAIL if any radius <= MIN_RADIUS."""
    base = evaluate_circle_packing_artifact(artifact)
    try:
        radii = np.asarray(artifact[1], dtype=float)
    except Exception:
        return base
    if radii.size and float(np.min(radii)) <= MIN_RADIUS:
        diag = EvaluationDiagnostics(
            constraints={"min_radius": MIN_RADIUS},
        )
        diag.violations.append(
            f"degenerate circle: min radius {float(np.min(radii)):.4g} <= {MIN_RADIUS} "
            "(every circle must have a strictly positive radius)"
        )
        return EvaluationResult(score=0.0, success=False, diagnostics=diag)
    return base


def biasfree_nz_domain():
    return DomainSpec(
        name="circle_packing",
        description=BARE_OBJECTIVE,
        initial_code=SEED_CODE,
        compiler=UvSandboxCompiler(
            allowed_imports=frozenset({"numpy", "math", "scipy", "itertools", "collections", "functools", "random", "heapq", "bisect"}),
            max_lines=None, timeout_seconds=TIMEOUT, seed=42),
        evaluator=evaluate_nonzero,
        allowed_imports=None, max_code_lines=None,
        hard_constraints=CONTRACT,
        examples=[SEED_CODE],
        hints=[],
        preferred_solution_shape=None,
    )
