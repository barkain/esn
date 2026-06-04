# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Elite and frontier archive managers for ESN core."""

from __future__ import annotations

from esn.core.models import CandidateRecord


class EliteArchive:
    """Stores valid, high-performing candidates with score-based eviction."""

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._candidates: list[CandidateRecord] = []

    @property
    def size(self) -> int:
        return len(self._candidates)

    def insert(self, candidate: CandidateRecord, min_score: float | None = None) -> bool:
        if not candidate.success or candidate.score is None:
            return False
        if min_score is not None and candidate.score < min_score:
            return False
        if len(self._candidates) >= self.max_size:
            # Evict lowest-score candidate
            min_idx = min(range(len(self._candidates)), key=lambda i: self._candidates[i].score)  # type: ignore[arg-type]
            if candidate.score <= self._candidates[min_idx].score:  # type: ignore[operator]
                return False
            self._candidates.pop(min_idx)
        self._candidates.append(candidate)
        return True

    def get_best(self, n: int = 1) -> list[CandidateRecord]:
        return sorted(self._candidates, key=lambda c: c.score or 0.0, reverse=True)[:n]

    def get_all(self) -> list[CandidateRecord]:
        return sorted(self._candidates, key=lambda c: c.score or 0.0, reverse=True)


class FrontierArchive:
    """Stores novel/repairable candidates without requiring validity."""

    def __init__(
        self,
        max_size: int = 100,
        novelty_threshold: float = 0.1,
        repairability_threshold: float = 0.1,
    ) -> None:
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold
        self.repairability_threshold = repairability_threshold
        self._candidates: list[CandidateRecord] = []
        self._novelty: dict[str, float] = {}
        self._repairability: dict[str, float] = {}
        self._insertion_order: list[str] = []

    @property
    def size(self) -> int:
        return len(self._candidates)

    def insert(
        self, candidate: CandidateRecord, novelty: float = 0.0, repairability: float = 0.0
    ) -> bool:
        if novelty < self.novelty_threshold and repairability < self.repairability_threshold:
            return False
        if len(self._candidates) >= self.max_size:
            self._evict()
        self._candidates.append(candidate)
        self._novelty[candidate.id] = novelty
        self._repairability[candidate.id] = repairability
        self._insertion_order.append(candidate.id)
        return True

    def _evict(self) -> None:
        # Evict oldest candidate with lowest novelty
        if not self._candidates:
            return
        min_novelty = min(self._novelty[c.id] for c in self._candidates)
        # Among those with lowest novelty, pick the oldest (earliest in insertion order)
        for cid in self._insertion_order:
            if self._novelty.get(cid) == min_novelty:
                self._candidates = [c for c in self._candidates if c.id != cid]
                self._insertion_order.remove(cid)
                del self._novelty[cid]
                del self._repairability[cid]
                return

    def get_repair_candidates(self, n: int = 5) -> list[CandidateRecord]:
        return sorted(
            self._candidates, key=lambda c: self._repairability.get(c.id, 0.0), reverse=True
        )[:n]

    def get_novel_candidates(self, n: int = 5) -> list[CandidateRecord]:
        return sorted(self._candidates, key=lambda c: self._novelty.get(c.id, 0.0), reverse=True)[
            :n
        ]

    def get_all(self) -> list[CandidateRecord]:
        return list(self._candidates)

    @property
    def distinct_object_hashes(self) -> int:
        return len({c.object_hash for c in self._candidates if c.object_hash})

    @property
    def novelty_scores(self) -> dict[str, float]:
        """Public read access to novelty scores by candidate id."""
        return dict(self._novelty)

    @property
    def repairability_scores(self) -> dict[str, float]:
        """Public read access to repairability scores by candidate id."""
        return dict(self._repairability)

    def set_scores(self, novelty: dict[str, float], repairability: dict[str, float]) -> None:
        """Restore novelty/repairability scores (used by persistence)."""
        self._novelty.update(novelty)
        self._repairability.update(repairability)
