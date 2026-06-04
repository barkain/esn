# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""LLM-backed Task 2 analyzer for ESN engine."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Protocol

from esn.engine.models import AnalysisResult, EvidenceItem, NewHypothesis


class LLMClient(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


class LLMAnalyzer:
    """Post-evaluation analyzer that emits evidence and new hypotheses."""

    def __init__(self, llm_client: LLMClient, *, max_new_hypotheses: int = 3) -> None:
        self._llm = llm_client
        self._max_new_hypotheses = max_new_hypotheses

    def analyze(
        self,
        solution_summary: str,
        score: float,
        diagnostics: Any,
        active_hypotheses: list[dict[str, Any]],
        strategy: str,
    ) -> AnalysisResult:
        try:
            response = self._llm(
                self._build_system_prompt(),
                self._build_user_prompt(
                    solution_summary, score, diagnostics, active_hypotheses, strategy
                ),
            )
            return self._parse_response(response, active_hypotheses)
        except Exception:
            return AnalysisResult()

    def _build_system_prompt(self) -> str:
        return (
            "You are an analyst for an evolutionary search system.\n"
            "Assign evidence for or against active hypotheses and propose a small number of new hypotheses.\n"
            'Return JSON: {"evidence": [{"hypothesis_id": "...", "evidence": 0|1, "explanation": "..."}], '
            '"new_hypotheses": [{"text": "...", "concepts": ["..."]}]}\n'
            "Prefer contrastive analysis and avoid restating the dominant story without new evidence.\n"
            "Output JSON only."
        )

    def _build_user_prompt(
        self,
        solution_summary: str,
        score: float,
        diagnostics: Any,
        active_hypotheses: list[dict[str, Any]],
        strategy: str,
    ) -> str:
        parts = [
            f"Solution summary: {solution_summary}",
            f"Score: {score}",
            f"Strategy: {strategy}",
            f"Diagnostics: {diagnostics}",
        ]
        if active_hypotheses:
            parts.append("Active hypotheses:")
            for hyp in active_hypotheses[:12]:
                parts.append(
                    f"- [{hyp.get('id', '?')}] (confidence={hyp.get('confidence', 'n/a')}): {hyp.get('text', '')}"
                )
        return "\n".join(parts)

    def _parse_response(
        self,
        response: str,
        active_hypotheses: list[dict[str, Any]],
    ) -> AnalysisResult:
        text = response.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid analyzer JSON: {exc}") from exc

        valid_ids = {h.get("id") for h in active_hypotheses}
        evidence: list[EvidenceItem] = []
        for item in data.get("evidence", []):
            hid = item.get("hypothesis_id", "")
            val = item.get("evidence")
            if hid not in valid_ids or val not in (0, 1):
                continue
            evidence.append(
                EvidenceItem(
                    hypothesis_id=hid,
                    evidence=int(val),
                    explanation=str(item.get("explanation", "")),
                )
            )

        new_hypotheses: list[NewHypothesis] = []
        for item in data.get("new_hypotheses", [])[: self._max_new_hypotheses]:
            text_val = str(item.get("text", "")).strip()
            if not text_val:
                continue
            concepts = item.get("concepts", [])
            if not isinstance(concepts, list):
                concepts = []
            new_hypotheses.append(NewHypothesis(text=text_val, concepts=[str(c) for c in concepts]))
        return AnalysisResult(evidence=evidence, new_hypotheses=new_hypotheses)
