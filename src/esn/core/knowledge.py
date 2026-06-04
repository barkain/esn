# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Knowledge bank integration for ESN core engine."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from esn.core.epistemic import update_hypothesis
from esn.core.knowledge_bank import KnowledgeBank
from esn.core.spectral_models import ESNConfig, HypothesisRecord

log = logging.getLogger(__name__)


class KnowledgeIntegration:
    """Manages hypothesis lifecycle for core engine.

    Wraps v1's KnowledgeBank and optional EmbeddingModel to:
    - Apply evidence from AnalysisResult to update hypothesis confidence
    - Add new hypotheses with proper embeddings
    - Provide formatted hypothesis lists for LLM prompts
    - Provide raw records for spectral pipeline
    """

    def __init__(
        self,
        config: ESNConfig | None = None,
        embedder: Any | None = None,  # EmbeddingModel from esn.core.embeddings
        embedding_dim: int | None = None,
    ) -> None:
        self._config = config or ESNConfig()
        self._embedder = embedder
        # Prefer embedder's actual dimension when available, so switching
        # embedding models via CLI doesn't silently desync spectral/zero
        # fallbacks from the real embedding shape.
        if embedding_dim is None:
            if embedder is not None and hasattr(embedder, "dimension"):
                embedding_dim = int(embedder.dimension)
            else:
                embedding_dim = int(self._config.embedding_dim)
        self._embedding_dim = embedding_dim
        self._bank = KnowledgeBank(self._config)

    @property
    def bank(self) -> KnowledgeBank:
        return self._bank

    def _admission_gate(self, candidate: HypothesisRecord) -> bool:
        """Check if candidate is sufficiently novel to admit.

        Uses cosine similarity + tag Jaccard overlap against active bank.
        Returns True if candidate should be admitted.
        """
        cos_thresh = self._config.admission_cosine_threshold  # default 0.88
        tag_thresh = self._config.admission_tag_overlap  # default 0.3

        candidate_emb = candidate.embedding
        c_norm = np.linalg.norm(candidate_emb)
        if c_norm < 1e-12:
            return True  # Zero embedding = can't compare, admit
        candidate_normed = candidate_emb / c_norm
        candidate_tags = {t.lower().replace("_", " ").strip() for t in candidate.concepts}

        for existing in self._bank.get_active_hypotheses():
            # Skip strongly refuted
            if existing.confidence < 0.2 and existing.n_obs >= 3:
                continue
            e_norm = np.linalg.norm(existing.embedding)
            if e_norm < 1e-12:
                continue
            cos = float(np.dot(candidate_normed, existing.embedding / e_norm))
            if cos < cos_thresh:
                continue
            # Check tag overlap
            existing_tags = {t.lower().replace("_", " ").strip() for t in existing.concepts}
            if candidate_tags and existing_tags:
                jac = len(candidate_tags & existing_tags) / len(candidate_tags | existing_tags)
            elif not candidate_tags and not existing_tags:
                jac = 1.0
            else:
                jac = 0.0
            if jac >= tag_thresh:
                log.debug(
                    "Admission gate rejected '%s' (cos=%.3f, jac=%.3f vs '%s')",
                    candidate.text[:60],
                    cos,
                    jac,
                    existing.text[:60],
                )
                return False  # Too similar, reject
        return True

    def process_analysis(
        self,
        analysis: Any,  # AnalysisResult from analyzer.py
        generation: int,
    ) -> dict:
        """Apply evidence and add new hypotheses.

        Mutates the knowledge bank. For novelty computation against the
        pre-update bank, use ``preview_analysis`` first.
        """
        prepared = self.preview_analysis(analysis, generation)
        self.apply_prepared_analysis(prepared, generation)
        return {
            "relevant_data": prepared["relevant_data"],
            "new_count": prepared["new_count"],
            "engaged": prepared["engaged"],
        }

    def preview_analysis(
        self,
        analysis: Any,  # AnalysisResult from analyzer.py
        generation: int,
        enrichment: str | None = None,
    ) -> dict:
        """Prepare analysis data without mutating the bank.

        Returns dict with keys needed for novelty computation:
        - relevant_data: list[dict] with {confidence, delta} per evidence item
        - new_count: number of new hypotheses added
        - engaged: list[HypothesisRecord] that were tested or created
        - updates: list of pending evidence updates for later application
        - new_records: list of pending new hypotheses for later admission
        """
        relevant_data = []
        engaged: list[HypothesisRecord] = []
        updates: list[dict] = []
        new_records: list[HypothesisRecord] = []

        # 1. Prepare evidence updates to existing hypotheses without mutating.
        for ev in analysis.evidence:
            hyp = self._bank.get(ev.hypothesis_id)
            if hyp is None or hyp.status != "active":
                continue
            c_old = hyp.confidence
            c_new, n_new, delta = update_hypothesis(hyp.confidence, hyp.n_obs, ev.evidence)
            relevant_data.append({"confidence": c_old, "delta": delta})
            engaged.append(hyp)
            updates.append(
                {
                    "hypothesis_id": hyp.id,
                    "confidence": c_new,
                    "n_obs": n_new,
                    "last_tested": generation,
                }
            )

        # 2. Prepare new hypotheses (with admission gate checked against current bank).
        # Phase 3.9: optionally append a multi-aspect enrichment tag to each
        # hypothesis text before embedding. The original text is preserved as
        # the first line — this is an additive change to what the embedder sees.
        from esn.core.observation_enrichment import enrich_hypothesis_text

        new_count = 0
        for new_hyp in analysis.new_hypotheses:
            enriched_text = enrich_hypothesis_text(new_hyp.text, enrichment or "")
            embedding = self._embed(enriched_text)
            record = HypothesisRecord(
                id=str(uuid.uuid4()),
                text=enriched_text,
                confidence=0.5,
                n_obs=1,
                embedding=embedding,
                concepts=new_hyp.concepts,
                created_at=generation,
                last_tested=generation,
                status="active",
            )
            if self._admission_gate(record):
                engaged.append(record)
                new_records.append(record)
                new_count += 1
            # else: silently dropped by admission gate

        return {
            "relevant_data": relevant_data,
            "new_count": new_count,
            "engaged": engaged,
            "updates": updates,
            "new_records": new_records,
        }

    def apply_prepared_analysis(self, prepared: dict, generation: int) -> None:
        """Apply a prepared analysis preview to the knowledge bank."""
        for update in prepared.get("updates", []):
            hyp = self._bank.get(update["hypothesis_id"])
            if hyp is None or hyp.status != "active":
                continue
            hyp.confidence = update["confidence"]
            hyp.n_obs = update["n_obs"]
            hyp.last_tested = update.get("last_tested", generation)

        for record in prepared.get("new_records", []):
            # Re-check admission to protect against edge cases where the bank
            # changed between preview and apply.
            if self._admission_gate(record):
                self._bank.add(record)

    def get_active_hypotheses_for_prompt(
        self,
        limit: int = 10,
        novelty_computer: Any = None,
    ) -> list[dict]:
        """Return active hypotheses formatted for LLM prompts.

        When ``novelty_computer`` is supplied AND it has actionable BBP spikes,
        this delegates to cluster-representative selection (Phase 3.10) so the
        prompt foregrounds the hypotheses that actually span the current
        principal directions of the knowledge bank. Otherwise it falls back to
        the legacy "sort by confidence distance from 0.5" behavior.
        """
        active = self._bank.get_active_hypotheses()

        if novelty_computer is not None:
            try:
                cluster_reps = novelty_computer.select_cluster_representatives(
                    active, per_cluster=2, limit=limit
                )
            except Exception:  # noqa: BLE001 - cluster selection is optional
                cluster_reps = None
            if cluster_reps:
                return [
                    {
                        "id": h.id,
                        "text": h.text,
                        "confidence": round(h.confidence, 3),
                        "source": "cluster",
                    }
                    for h in cluster_reps
                ]

        # Fallback: sort by confidence distance from 0.5 (most certain first)
        sorted_hyps = sorted(active, key=lambda h: abs(h.confidence - 0.5), reverse=True)
        return [
            {"id": h.id, "text": h.text, "confidence": round(h.confidence, 3)}
            for h in sorted_hyps[:limit]
        ]

    def get_active_hypothesis_records(self) -> list[HypothesisRecord]:
        """Return raw HypothesisRecord objects for spectral pipeline."""
        return self._bank.get_active_hypotheses()

    def run_maintenance(self, generation: int) -> dict:
        """Run hypothesis lifecycle maintenance.

        Orchestrates: dedup -> TTL/retire -> archive old retired.
        Returns stats dict.
        """
        stats: dict[str, int] = {"deduped": 0, "retired": 0, "archived": 0}

        # 1. Dedup (if enough hypotheses)
        active = self._bank.get_active_hypotheses()
        if len(active) >= 4:  # Need minimum for meaningful dedup
            from esn.core.dedup import DedupConfig, apply_dedup, run_dedup

            dedup_config = DedupConfig(
                tag_jaccard_threshold=self._config.admission_tag_overlap,
            )
            result = run_dedup(active, dedup_config)
            if result.total_merged > 0:
                stats["deduped"] = apply_dedup(active, result)
                self._bank._invalidate_cache()

        # 2. TTL + low-confidence retirement
        before_retired = sum(1 for h in self._bank.hypotheses if h.status == "retired")
        self._bank.retire_hypotheses(current_generation=generation)
        after_retired = sum(1 for h in self._bank.hypotheses if h.status == "retired")
        stats["retired"] = after_retired - before_retired

        # 3. Archive old retired (keep max 50 retired)
        retired = [h for h in self._bank.get_all_hypotheses() if h.status == "retired"]
        max_retired = 50
        if len(retired) > max_retired:
            retired.sort(key=lambda h: h.last_tested)
            for h in retired[: len(retired) - max_retired]:
                h.status = "archived"
                stats["archived"] += 1
            self._bank._invalidate_cache()

        stats["active"] = self._bank.active_count()
        stats["total"] = self._bank.size()

        log.debug(
            "Maintenance gen=%d: deduped=%d retired=%d archived=%d active=%d",
            generation,
            stats["deduped"],
            stats["retired"],
            stats["archived"],
            stats["active"],
        )
        return stats

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using embedder if available, else zero vector."""
        if self._embedder is not None:
            return self._embedder.embed(text)
        return np.zeros(self._embedding_dim)
