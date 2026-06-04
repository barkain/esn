# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Protocol for deterministic post-mutation local optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class TunableFamily:
    """Minimal shape for a tunable solution family.

    Captures enough structure for deterministic parameter optimization
    without requiring the optimizer to understand the domain.
    """

    family_name: str  # e.g. "grid_5x5_plus_one"
    params: dict[str, float]  # current parameter values
    bounds: dict[str, tuple[float, float]]  # parameter bounds (lo, hi)
    score: float = 0.0  # current score with these params


@dataclass
class LocalImprovementResult:
    """Result of a local improvement attempt."""

    improved: bool  # whether improvement was found
    code: str  # new program code (may be original if not improved)
    artifact: Any = None  # new artifact (may be None if not improved)
    score: float = 0.0  # new score (0 if not improved)
    improvement_delta: float = 0.0  # score - original_score
    steps_taken: int = 0  # number of optimization steps
    method: str = ""  # description of what was done


@runtime_checkable
class LocalImprover(Protocol):
    """Deterministic post-mutation optimizer. Improves a solution without LLM calls."""

    def improve(
        self,
        code: str,
        artifact: Any,
        score: float,
        evaluator: Any,
    ) -> LocalImprovementResult: ...
