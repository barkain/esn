# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Core data models for ESN core search architecture."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field  # type: ignore[import-not-found]

from esn.core.enums import SearchMode  # type: ignore[import-not-found]


class CompilerResult(BaseModel):
    artifact: Any
    success: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FragmentInterface(BaseModel):
    """Interface contract for a bounded editable fragment."""

    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    pure: bool = True
    function_name: str = ""


class Fragment(BaseModel):
    """Named, typed, size-bounded artifact stored inside a SearchObject."""

    name: str
    code: str
    interface: FragmentInterface
    constraints: list[str] = Field(default_factory=list)
    max_lines: int = 30


class EvaluationDiagnostics(BaseModel):
    constraints: dict[str, Any] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    residuals: dict[str, float] = Field(default_factory=dict)
    complexity: dict[str, float] = Field(default_factory=dict)
    robustness: dict[str, float] = Field(default_factory=dict)
    resources: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """What a ``DomainSpec.evaluator`` must return for each candidate.

    ``score`` is higher-is-better (the engine maximizes it); ``success`` gates
    whether the candidate is eligible to become the run's best (a ``False``
    result is recorded but never promoted, whatever its score).
    """

    score: float
    success: bool
    diagnostics: EvaluationDiagnostics | None = None
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class MutationPlan(BaseModel):
    search_mode: SearchMode
    operator_name: str
    target: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    expected_effect: str = ""
    risk: str = "low"


class MutationContext(BaseModel):
    search_mode: SearchMode
    parent_summary: str = ""
    diagnostics: EvaluationDiagnostics | None = None
    top_hypotheses: list[str] = Field(default_factory=list)
    spectral_guidance: dict[str, Any] = Field(default_factory=dict)


class MutationResult(BaseModel):
    mutated_object: Any = None
    success: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImprovementContext(BaseModel):
    search_mode: SearchMode
    diagnostics: EvaluationDiagnostics | None = None
    budget: int | float = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImprovementResult(BaseModel):
    improved_object: Any = None
    success: bool
    changed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class CandidateRecord(BaseModel):
    id: str
    generation: int
    parent_id: str | None = None
    search_mode: SearchMode
    operator_name: str
    object_hash: str
    object_summary: str = ""
    score: float | None = None
    success: bool | None = None
    diagnostics: EvaluationDiagnostics | None = None
    epistemic_novelty: float | None = None
    spectral_novelty: float | None = None
    plan_rationale: str = ""
    plan_expected_effect: str = ""
    compiled_artifact: str = ""
    realized_artifact_summary: str = ""
    family: str = ""  # coarse structural bucket (recursive-multi, iterative-nested, ...)
    family_confidence: str = "none"  # "high", "medium", "low", "none"
    aspect_signature: str = ""  # "family=X | features=a,b,c | cfhash=abcd1234"
    slot: int | None = None
    branch_id: str | None = None  # branch preservation
    compile_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class OperatorStats(BaseModel):
    attempts: int = 0
    compile_successes: int = 0
    eval_successes: int = 0
    mean_score_delta: float = 0.0
    recent_score_delta: float = 0.0
    mean_epistemic_novelty: float = 0.0
    mean_spectral_novelty: float = 0.0
    non_improving_streak: int = 0
    last_used_generation: int = 0


class SearchState(BaseModel):
    generation: int = 0
    best_score: float = 0.0
    recent_scores: list[float] = Field(default_factory=list)
    recent_operators: list[str] = Field(default_factory=list)
    stagnation_counter: int = 0
    current_mode: SearchMode = SearchMode.EXPLOIT
    elite_size: int = 0
    frontier_size: int = 0
    frontier_distinct_count: int = 0
