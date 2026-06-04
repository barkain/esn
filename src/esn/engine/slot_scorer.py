# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Batch slot scorer: feature-based scoring + MMR diverse selection for parent/style allocation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from esn.core.enums import SearchMode

logger = logging.getLogger(__name__)

# Mutation style mapping per search mode (mirrored from engine)
_MODE_STYLE_MAP: dict[SearchMode, list[str]] = {
    SearchMode.EXPLOIT: ["refine", "explore"],
    SearchMode.EXPLORE: ["explore", "radical"],
    SearchMode.REPAIR: ["repair", "refine"],
    SearchMode.BRIDGE: ["synthesize"],
    SearchMode.RECOVER: ["repair", "refine"],
    SearchMode.COMPRESS: ["refine"],
}


@dataclass
class ScorerWeights:
    """Configurable weights for slot scoring."""

    # Quality weights
    w_parent_quantile: float = 0.25
    w_parent_gap: float = -0.15
    w_style_success: float = 0.20
    w_style_delta: float = 0.15
    # Diversity weights
    w_family_rarity: float = 0.25
    w_combo_recency: float = 0.20
    w_style_novelty: float = 0.15
    w_combo_untried_bonus: float = 0.30
    # Penalty weights
    w_style_streak: float = -0.10
    w_family_plateau: float = -0.15
    w_parent_breakthrough: float = 0.20
    # Blend
    stagnation_diversity_scale: float = 10.0
    # MMR
    mmr_lambda: float = 0.7


@dataclass
class ScorerState:
    """Engine state snapshot for scoring."""

    generation: int = 0
    best_score: float = 0.0
    stagnation_counter: int = 0
    consecutive_failures: int = 0
    elite_size: int = 0
    frontier_size: int = 0
    num_families: int = 0
    spectral_gamma: float = 0.0
    spectral_num_spikes: int = 0
    recent_improvement_rate: float = 0.0


@dataclass
class SlotCandidate:
    """A proposed (parent, style) pair for batch allocation."""

    parent_id: str
    parent_score: float
    parent_family: str
    style: str
    features: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class ComboTracker:
    """Lightweight (family, style) combo counter."""

    _counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    _successes: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    _last_gen: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def record(self, family: str, style: str, success: bool, generation: int) -> None:
        """Record a combo outcome."""
        key = (family, style)
        self._counts[key] += 1
        if success:
            self._successes[key] += 1
        self._last_gen[key] = generation

    def get(self, family: str, style: str) -> tuple[int, int, int]:
        """Returns (attempts, successes, last_gen)."""
        key = (family, style)
        return self._counts[key], self._successes[key], self._last_gen[key]

    def total_attempts(self) -> int:
        """Total attempts across all combos."""
        return sum(self._counts.values())

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "counts": {f"{f}::{s}": v for (f, s), v in self._counts.items()},
            "successes": {f"{f}::{s}": v for (f, s), v in self._successes.items()},
            "last_gen": {f"{f}::{s}": v for (f, s), v in self._last_gen.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ComboTracker:
        """Deserialize from persistence."""
        tracker = cls()
        for key_str, v in data.get("counts", {}).items():
            parts = key_str.split("::", 1)
            if len(parts) == 2:
                tracker._counts[(parts[0], parts[1])] = v
        for key_str, v in data.get("successes", {}).items():
            parts = key_str.split("::", 1)
            if len(parts) == 2:
                tracker._successes[(parts[0], parts[1])] = v
        for key_str, v in data.get("last_gen", {}).items():
            parts = key_str.split("::", 1)
            if len(parts) == 2:
                tracker._last_gen[(parts[0], parts[1])] = v
        return tracker


class BatchSlotScorer:
    """Feature-based scoring + MMR diverse selection for batch slot allocation."""

    def __init__(self, weights: ScorerWeights | None = None) -> None:
        self.weights = weights or ScorerWeights()
        self.combo_tracker = ComboTracker()

    def snapshot_state(self, engine: Any) -> ScorerState:
        """Extract generic state from engine. No domain coupling."""
        # Spectral info
        spectral_gamma = 0.0
        spectral_num_spikes = 0
        if engine.novelty_computer:
            ss = getattr(engine.novelty_computer, "spectral_state", None)
            if ss is not None:
                spectral_gamma = getattr(ss, "gamma_t", 0.0)
                spectral_num_spikes = getattr(ss, "num_spikes", 0)

        # Recent improvement rate: fraction of last 8 gens that improved
        recent_scores = list(engine.state.recent_scores)[-8:]
        improvements = 0
        for i in range(1, len(recent_scores)):
            if recent_scores[i] > recent_scores[i - 1]:
                improvements += 1
        rate = improvements / max(1, len(recent_scores) - 1) if len(recent_scores) > 1 else 0.0

        return ScorerState(
            generation=engine.generation,
            best_score=engine._best_score,
            stagnation_counter=engine.state.stagnation_counter,
            consecutive_failures=engine._consecutive_failures,
            elite_size=engine.elite_archive.size,
            frontier_size=engine.frontier_archive.size,
            num_families=len(engine._family_tracker._families),
            spectral_gamma=spectral_gamma,
            spectral_num_spikes=spectral_num_spikes,
            recent_improvement_rate=rate,
        )

    def enumerate_candidates(
        self, engine: Any, mode: SearchMode
    ) -> tuple[list[SlotCandidate], dict[str, str]]:
        """Enumerate all (parent, style) combos. Returns candidates + parent_id->code map."""
        parents: list[tuple[str, str, float, str]] = []  # (id, code, score, family)
        seen_hashes: set[str] = set()

        def _add_parent(code: str, score: float, family: str) -> None:
            h = hashlib.sha256(code.encode()).hexdigest()[:16]
            if h not in seen_hashes:
                seen_hashes.add(h)
                parents.append((h, code, score, family))

        # 1. Best code (always)
        best_family = self._get_best_family(engine)
        _add_parent(engine._best_code, engine._best_score, best_family)

        # 2. Top 3 elites
        for elite in engine.elite_archive.get_best(3):
            code = engine._program_store.get(elite.id, "")
            if code:
                family = elite.family or "unknown"
                _add_parent(code, elite.score or 0.0, family)

        # 3. Top 2 frontier novel candidates
        if engine.frontier_archive.size > 0:
            for novel in engine.frontier_archive.get_novel_candidates(2):
                code = engine._program_store.get(novel.id, "")
                if code:
                    family = novel.family or "unknown"
                    _add_parent(code, novel.score or 0.0, family)

        # Ensure at least one parent
        if not parents:
            h = hashlib.sha256(engine._best_code.encode()).hexdigest()[:16]
            parents.append((h, engine._best_code, engine._best_score, "unknown"))

        # Build code map
        code_map: dict[str, str] = {pid: code for pid, code, _, _ in parents}

        # Styles for this mode
        styles = _MODE_STYLE_MAP.get(mode, ["refine"])

        # Cross product
        candidates: list[SlotCandidate] = []
        for pid, _code, pscore, pfamily in parents:
            for style in styles:
                candidates.append(
                    SlotCandidate(
                        parent_id=pid,
                        parent_score=pscore,
                        parent_family=pfamily,
                        style=style,
                    )
                )

        return candidates, code_map

    def featurize(
        self,
        candidate: SlotCandidate,
        state: ScorerState,
        engine: Any,
        all_parent_scores: list[float],
    ) -> None:
        """Compute all features for a candidate. Mutates features dict in place."""
        f = candidate.features

        # --- Parent features ---
        # parent_score_quantile: rank among enumerated parents, best=1.0
        unique_scores = sorted(set(all_parent_scores))
        if len(unique_scores) <= 1:
            f["parent_score_quantile"] = 1.0
        else:
            rank = unique_scores.index(candidate.parent_score)
            f["parent_score_quantile"] = rank / (len(unique_scores) - 1)

        # parent_gap_to_best
        if state.best_score > 0:
            f["parent_gap_to_best"] = max(
                0.0, min(1.0, (state.best_score - candidate.parent_score) / state.best_score)
            )
        else:
            f["parent_gap_to_best"] = 0.0

        # parent_is_best
        f["parent_is_best"] = 1.0 if candidate.parent_score == state.best_score else 0.0

        # parent_recent_breakthrough: 1.0 if family improved in last 3 gens
        family_stats = engine._family_tracker.get_stats(candidate.parent_family)
        if family_stats and family_stats.plateau_gens <= 3:
            f["parent_recent_breakthrough"] = 1.0
        else:
            f["parent_recent_breakthrough"] = 0.0

        # --- Family features ---
        # family_rarity: 1 - (family_archive_count / total_archive_size)
        total_archive = engine.elite_archive.size + engine.frontier_archive.size
        if total_archive > 0:
            family_count = sum(
                1 for c in engine.elite_archive.get_all() if c.family == candidate.parent_family
            ) + sum(
                1 for c in engine.frontier_archive.get_all() if c.family == candidate.parent_family
            )
            f["family_rarity"] = max(0.0, min(1.0, 1.0 - family_count / total_archive))
        else:
            f["family_rarity"] = 1.0

        # family_plateau_gens
        if family_stats:
            f["family_plateau_gens"] = min(
                1.0, family_stats.plateau_gens / max(10, state.generation)
            )
        else:
            f["family_plateau_gens"] = 0.0

        # --- Style features ---
        style_stats = engine.credit_model.get_stats(candidate.style)

        f["style_success_rate"] = style_stats.eval_successes / max(1, style_stats.attempts)

        # style_recent_delta_z: normalized by max abs across all styles
        all_style_stats = engine.credit_model.get_all_stats()
        all_deltas = [s.recent_score_delta for s in all_style_stats.values()]
        max_abs_delta = max((abs(d) for d in all_deltas), default=0.0)
        if max_abs_delta > 0:
            f["style_recent_delta_z"] = max(
                -1.0, min(1.0, style_stats.recent_score_delta / max_abs_delta)
            )
        else:
            f["style_recent_delta_z"] = 0.0

        f["style_novelty_yield"] = style_stats.mean_epistemic_novelty

        f["style_non_improving_frac"] = min(
            1.0, style_stats.non_improving_streak / max(1, style_stats.attempts)
        )

        total_style_attempts = sum(s.attempts for s in all_style_stats.values())
        f["style_attempts_share"] = style_stats.attempts / max(1, total_style_attempts)

        # --- Combo features ---
        combo_attempts, combo_successes, combo_last_gen = self.combo_tracker.get(
            candidate.parent_family, candidate.style
        )
        total_combo = self.combo_tracker.total_attempts()

        f["combo_attempt_count"] = combo_attempts / max(1, total_combo)

        f["combo_success_rate"] = combo_successes / max(1, combo_attempts)

        # combo_recency: min(1.0, (generation - last_gen) / 5), 1.0 if never used
        if combo_attempts == 0:
            f["combo_recency"] = 1.0
        else:
            f["combo_recency"] = min(1.0, (state.generation - combo_last_gen) / 5)

    def score_candidates(self, candidates: list[SlotCandidate], state: ScorerState) -> None:
        """Score all candidates. Mutates score field in place."""
        for c in candidates:
            f = c.features

            quality = (
                self.weights.w_parent_quantile * f["parent_score_quantile"]
                + self.weights.w_style_success * f["style_success_rate"]
                + self.weights.w_style_delta * f["style_recent_delta_z"]
                + self.weights.w_parent_breakthrough * f["parent_recent_breakthrough"]
            )

            diversity = (
                self.weights.w_family_rarity * f["family_rarity"]
                + self.weights.w_combo_recency * f["combo_recency"]
                + self.weights.w_style_novelty * f["style_novelty_yield"]
                + self.weights.w_combo_untried_bonus
                * (1.0 if f["combo_attempt_count"] == 0 else 0.0)
            )

            penalty = (
                self.weights.w_style_streak * f["style_non_improving_frac"]
                + self.weights.w_family_plateau * f["family_plateau_gens"]
                + self.weights.w_parent_gap * f["parent_gap_to_best"]
            )

            alpha = min(1.0, state.stagnation_counter / self.weights.stagnation_diversity_scale)
            c.score = (1 - alpha) * quality + alpha * diversity + penalty

    def select(self, scored: list[SlotCandidate], k: int) -> list[SlotCandidate]:
        """MMR-style greedy diverse selection."""
        scored_sorted = sorted(scored, key=lambda c: c.score, reverse=True)
        if not scored_sorted:
            return []

        selected = [scored_sorted[0]]
        remaining = list(scored_sorted[1:])

        lam = self.weights.mmr_lambda
        while len(selected) < k and remaining:
            best_mmr = -float("inf")
            best_idx = 0
            for i, c in enumerate(remaining):
                sim = self._max_similarity(c, selected)
                mmr = lam * c.score - (1 - lam) * sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    def _max_similarity(self, c: SlotCandidate, selected: list[SlotCandidate]) -> float:
        """Max similarity to any already-selected candidate. Returns [0, 1]."""
        sim = 0.0
        for s in selected:
            s_val = 0.0
            if c.parent_id == s.parent_id:
                s_val += 0.4
            if c.parent_family == s.parent_family:
                s_val += 0.3
            if c.style == s.style:
                s_val += 0.3
            sim = max(sim, s_val)
        return sim

    def plan_batch(self, engine: Any, mode: SearchMode) -> list[tuple[str, str]]:
        """Top-level entry point. Returns [(parent_code, style), ...].

        Slot 0 is always reserved as an exploitation anchor: (best_code, "refine").
        The remaining k-1 slots are filled by the scorer pipeline with MMR selection.
        """
        k = engine._batch_size

        # --- Slot 0: exploitation anchor ---
        anchor_code = engine._best_code
        anchor_style = "refine"

        if k <= 1:
            return [(anchor_code, anchor_style)]

        state = self.snapshot_state(engine)
        candidates, code_map = self.enumerate_candidates(engine, mode)

        if not candidates:
            return [(anchor_code, anchor_style)]

        # Determine anchor parent_id so we can exclude duplicates
        anchor_parent_id = hashlib.sha256(anchor_code.encode()).hexdigest()[:16]

        # Remove candidates that duplicate the anchor (same parent + refine)
        candidates = [
            c
            for c in candidates
            if not (c.parent_id == anchor_parent_id and c.style == anchor_style)
        ]

        all_parent_scores = [c.parent_score for c in candidates]
        for c in candidates:
            self.featurize(c, state, engine, all_parent_scores)

        self.score_candidates(candidates, state)
        remaining_k = k - 1
        selected = self.select(candidates, remaining_k)

        self._log_decisions(
            engine, state, candidates, selected, anchor=(anchor_parent_id, anchor_style)
        )

        result = [(anchor_code, anchor_style)]
        result.extend((code_map[s.parent_id], s.style) for s in selected)

        # Pad with cycling if fewer unique candidates than batch size
        if len(result) < k and len(result) > 1:
            pool = result[1:]  # cycle over non-anchor slots
            while len(result) < k:
                result.append(pool[(len(result) - 1) % len(pool)])

        return result

    def record_outcome(self, family: str, style: str, success: bool, generation: int) -> None:
        """Record a combo outcome. Called from engine._process_outcome."""
        self.combo_tracker.record(family, style, success, generation)

    def _get_best_family(self, engine: Any) -> str:
        """Get family of the current best code."""
        # Check family tracker for the most recent family with best score
        best_family = "unknown"
        best_score = 0.0
        for fs in engine._family_tracker._families.values():
            if fs.best_score >= best_score:
                best_score = fs.best_score
                best_family = fs.name
        return best_family

    def _log_decisions(
        self,
        engine: Any,
        state: ScorerState,
        all_candidates: list[SlotCandidate],
        selected: list[SlotCandidate],
        anchor: tuple[str, str] | None = None,
    ) -> None:
        """Log scorer decisions to scorer_decisions.jsonl in run directory."""
        run_dir = getattr(engine, "_run_dir", None)
        if not run_dir:
            return

        run_dir = Path(run_dir)
        if not run_dir.exists():
            return

        k = len(selected)
        selected_ids = {(s.parent_id, s.style) for s in selected}

        # Top 2k candidates: selected + best rejected
        rejected = [
            c
            for c in sorted(all_candidates, key=lambda c: c.score, reverse=True)
            if (c.parent_id, c.style) not in selected_ids
        ][:k]

        alpha = min(1.0, state.stagnation_counter / self.weights.stagnation_diversity_scale)

        entry: dict[str, Any] = {
            "generation": state.generation,
            "state": asdict(state),
            "alpha": alpha,
        }

        # Record the exploitation anchor (slot 0)
        if anchor is not None:
            entry["anchor"] = {
                "parent_id": anchor[0],
                "style": anchor[1],
                "slot": 0,
                "reason": "exploitation_anchor",
            }

        entry["selected"] = [
            {
                "parent_id": s.parent_id,
                "parent_family": s.parent_family,
                "style": s.style,
                "score": s.score,
                "features": s.features,
            }
            for s in selected
        ]
        entry["top_rejected"] = [
            {
                "parent_id": r.parent_id,
                "parent_family": r.parent_family,
                "style": r.style,
                "score": r.score,
                "features": r.features,
            }
            for r in rejected
        ]

        log_path = run_dir / "scorer_decisions.jsonl"
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.debug("Failed to write scorer decisions log")

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "weights": asdict(self.weights),
            "combo_tracker": self.combo_tracker.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BatchSlotScorer:
        """Deserialize from persistence."""
        weights_data = data.get("weights", {})
        weights = ScorerWeights(**weights_data)
        scorer = cls(weights=weights)
        combo_data = data.get("combo_tracker")
        if combo_data:
            scorer.combo_tracker = ComboTracker.from_dict(combo_data)
        return scorer
