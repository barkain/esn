# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Protocol definitions for ESN engine program-level search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from esn.engine.models import (
        AnalysisResult,
        MutationContext,
        MutationResult,
        PredictionResult,
    )

from esn.core.models import CompilerResult


@runtime_checkable
class ProgramObject(Protocol):
    """A search object that IS a program / solver / procedure.

    The program is executable code that, when compiled (executed in a sandbox),
    produces a candidate artifact for evaluation.
    """

    @property
    def code(self) -> str:
        """The program source code."""
        ...

    def summary(self) -> str:
        """Human-readable description of the program's strategy."""
        ...

    def structural_hash(self) -> str:
        """Content hash of the code (SHA-256 of normalized source)."""
        ...

    def serialize(self) -> str:
        """Serialize for persistence (JSON string)."""
        ...

    @classmethod
    def deserialize(cls, data: str) -> ProgramObject:
        """Deserialize from persistence."""
        ...


@runtime_checkable
class ProgramCompiler(Protocol):
    """Compile (execute) program code in a sandboxed environment."""

    def compile(self, code: str, seed: int = 42) -> CompilerResult:
        """Execute program code in a sandboxed environment."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """Task 1: Pre-evaluation prediction of score range and hypothesis relevance."""

    def predict(
        self,
        program: ProgramObject,
        mutation_style: str,
        hypotheses: list[dict[str, Any]],
        score_history: dict[str, Any],
    ) -> PredictionResult: ...


@runtime_checkable
class Mutator(Protocol):
    """LLM-driven program mutation with style-specific prompting."""

    def mutate(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult: ...


@runtime_checkable
class Analyzer(Protocol):
    """Task 2: Post-evaluation analysis producing evidence and new hypotheses."""

    def analyze(
        self,
        solution_summary: str,
        score: float,
        diagnostics: Any,
        active_hypotheses: list[dict[str, Any]],
        strategy: str,
    ) -> AnalysisResult: ...
