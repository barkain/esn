# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""LLM-backed Task 1 predictor for ESN engine."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Protocol

from esn.engine.models import PredictionResult
from esn.engine.protocols import ProgramObject


class LLMClient(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


class LLMPredictor:
    """Cheap-model predictor for score range and hypothesis relevance."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def predict(
        self,
        program: ProgramObject,
        mutation_style: str,
        hypotheses: list[dict[str, Any]],
        score_history: dict[str, Any],
    ) -> PredictionResult:
        try:
            response = self._llm(
                self._build_system_prompt(),
                self._build_user_prompt(program, mutation_style, hypotheses, score_history),
            )
            return self._parse_response(response, hypotheses)
        except Exception:
            return PredictionResult()

    def _build_system_prompt(self) -> str:
        return (
            "You are a fast prediction model in an evolutionary search system.\n"
            "Given a candidate program and current score history, estimate a plausible score range "
            "and identify which hypotheses are most relevant.\n"
            'Return JSON: {"score_range": [low, high], "relevant_hypothesis_ids": ["..."], "reasoning": "..."}\n'
            "Output JSON only."
        )

    def _build_user_prompt(
        self,
        program: ProgramObject,
        mutation_style: str,
        hypotheses: list[dict[str, Any]],
        score_history: dict[str, Any],
    ) -> str:
        parts = [
            f"Mutation style: {mutation_style}",
            f"Score history: {score_history}",
            "Program:",
            program.code,
        ]
        if hypotheses:
            parts.append("Hypotheses:")
            for hyp in hypotheses[:10]:
                parts.append(f"- [{hyp.get('id', '?')}] {hyp.get('text', '')}")
        return "\n".join(parts)

    def _parse_response(
        self,
        response: str,
        hypotheses: list[dict[str, Any]],
    ) -> PredictionResult:
        text = response.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid predictor JSON: {exc}") from exc

        score_range = data.get("score_range", [0.0, 1.0])
        if not isinstance(score_range, list | tuple) or len(score_range) != 2:
            score_range = [0.0, 1.0]
        lo, hi = float(score_range[0]), float(score_range[1])
        if lo > hi:
            lo, hi = hi, lo
        valid_ids = {h.get("id") for h in hypotheses}
        relevant = [hid for hid in data.get("relevant_hypothesis_ids", []) if hid in valid_ids]
        return PredictionResult(
            score_range=(lo, hi),
            relevant_hypothesis_ids=relevant,
            reasoning=str(data.get("reasoning", "")),
        )
