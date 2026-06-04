# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Knowledge bank for managing hypothesis records."""

from .spectral_models import HypothesisRecord, HypothesisStatus, ESNConfig


class KnowledgeBank:
    """Manages the collection of hypothesis records."""

    def __init__(self, config: ESNConfig) -> None:
        self.config = config
        self.hypotheses: list[HypothesisRecord] = []
        self._active_cache: list[HypothesisRecord] | None = None

    def _invalidate_cache(self) -> None:
        self._active_cache = None

    def add(self, hypothesis: HypothesisRecord) -> None:
        """Add a new hypothesis to the bank."""
        self.hypotheses.append(hypothesis)
        self._invalidate_cache()

    def get_active_hypotheses(self) -> list[HypothesisRecord]:
        """Get all active hypotheses (cached until mutation)."""
        if self._active_cache is None:
            self._active_cache = [h for h in self.hypotheses if h.status == HypothesisStatus.ACTIVE]
        return self._active_cache

    def get_all_hypotheses(self) -> list[HypothesisRecord]:
        """Get all hypotheses regardless of status."""
        return self.hypotheses

    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        """Look up a hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None

    def active_count(self) -> int:
        """Return count of active hypotheses."""
        return len(self.get_active_hypotheses())

    def size(self) -> int:
        """Return total number of hypotheses."""
        return len(self.hypotheses)

    def retire_hypotheses(self, current_generation: int = 0) -> None:
        """Retire hypotheses that meet criteria.

        Retirement triggers:
        1. Low confidence: c < threshold AND n_obs >= min_obs
        2. TTL expiry: n_obs == 1 (never tested) AND age > hypothesis_ttl gens
        """
        changed = False
        ttl = self.config.hypothesis_ttl
        for hyp in self.hypotheses:
            if hyp.status != HypothesisStatus.ACTIVE:
                continue
            # Low-confidence retirement
            if (
                hyp.confidence < self.config.retirement_threshold
                and hyp.n_obs >= self.config.retirement_min_obs
            ):
                hyp.status = HypothesisStatus.RETIRED
                changed = True
            # TTL: never-tested hypothesis expired
            elif ttl > 0 and hyp.n_obs == 1 and current_generation - hyp.created_at >= ttl:
                hyp.status = HypothesisStatus.RETIRED
                changed = True
        if changed:
            self._invalidate_cache()
