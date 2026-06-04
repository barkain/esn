# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Data models for ESN engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Evidence for/against a hypothesis."""

    hypothesis_id: str
    evidence: int  # 1 = support, 0 = contradict
    explanation: str = ""


class NewHypothesis(BaseModel):
    """A new hypothesis proposed by the analyzer."""

    text: str
    concepts: list[str] = Field(default_factory=list)


class PredictionResult(BaseModel):
    """Task 1 output: pre-evaluation prediction."""

    score_range: tuple[float, float] = (0.0, 1.0)
    relevant_hypothesis_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class MutationContext(BaseModel):
    """Context passed to the mutator for informed code generation."""

    search_mode: str = "exploit"
    mutation_style: str = "refine"
    top_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    spectral_guidance: dict[str, Any] = Field(default_factory=dict)
    search_temperature: float = 0.0
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    score_history: dict[str, Any] = Field(default_factory=dict)
    error_context: str = ""
    targeted_hypothesis_ids: list[str] = Field(default_factory=list)
    intended_effect: str = ""

    # Global search narrative fields (for explore/radical prompts)
    best_code: str = ""
    best_score: float = 0.0
    recent_attempts: list[dict[str, Any]] = Field(default_factory=list)
    archive_families: list[str] = Field(default_factory=list)
    stagnation_gens: int = 0

    # Family reasoning
    family_summaries: list[str] = Field(default_factory=list)
    parent_family: str = ""
    family_failure_reasons: dict[str, list[str]] = Field(default_factory=dict)
    # family -> list of recent failure reasons


class MutationResult(BaseModel):
    """Output from the mutator."""

    code: str = ""
    success: bool = False
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Task 2 output: evidence and new hypotheses."""

    evidence: list[EvidenceItem] = Field(default_factory=list)
    new_hypotheses: list[NewHypothesis] = Field(default_factory=list)
