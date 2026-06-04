# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Adaptive batch budget controller for ESN engine.

Tracks a run-level slot budget and computes per-generation batch approvals
using deterministic heuristics.  Phase 3 (full adaptive enforcement): the
engine can both shrink batch below nominal and expand above it.  Expansion
requires passing an explicit gate (multiple branches, family diversity,
positive extra-slot yield, low waste, budget pace under control).

Design intent: adaptive, budgeted mutator batch control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchBudgetConfig:
    """Tuning knobs for the batch budget controller."""

    max_batch_size: int = 8
    """Hard upper bound for any single generation's batch."""

    lookback_window: int = 5
    """Number of recent generations used for yield / diversity stats."""

    # Marginal-cost tier multipliers (slot index → multiplier).
    # Slots 1-3 are cheap; 4-5 moderate; 6+ expensive.
    tier_boundaries: tuple[int, ...] = (4, 6)
    tier_multipliers: tuple[float, ...] = (1.0, 1.5, 2.5)

    # Threshold for "meaningful" branch
    meaningful_branch_min_depth: int = 2
    meaningful_branch_min_score_fraction: float = 0.3


# ---------------------------------------------------------------------------
# Per-generation yield record
# ---------------------------------------------------------------------------


@dataclass
class GenerationYield:
    """What a single generation's batch actually produced."""

    generation: int
    batch_size: int
    successes: int = 0
    frontier_improvements: int = 0
    best_improved: bool = False
    unique_families: int = 0
    collapsed_count: int = 0
    duplicate_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.batch_size if self.batch_size else 0.0

    @property
    def extra_slot_yield(self) -> float:
        """Fraction of slots beyond the first that produced a success."""
        extra = self.batch_size - 1
        extra_successes = max(0, self.successes - 1)
        return extra_successes / extra if extra > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "batch_size": self.batch_size,
            "successes": self.successes,
            "frontier_improvements": self.frontier_improvements,
            "best_improved": self.best_improved,
            "unique_families": self.unique_families,
            "collapsed_count": self.collapsed_count,
            "duplicate_count": self.duplicate_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationYield:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------------------
# Shadow approval decision
# ---------------------------------------------------------------------------


@dataclass
class BatchDecision:
    """Result of the batch approval algorithm."""

    requested: int
    approved: int
    actual: int  # what the engine will really use (min of nominal and approved in Phase 2)
    fair_share: float
    pace_ratio: float
    slots_remaining: int
    marginal_trace: list[dict[str, float]] = field(default_factory=list)
    """Per-slot marginal value and cost that led to the decision."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "approved": self.approved,
            "actual": self.actual,
            "fair_share": round(self.fair_share, 2),
            "pace_ratio": round(self.pace_ratio, 3),
            "slots_remaining": self.slots_remaining,
            "marginal_trace": self.marginal_trace,
        }


# ---------------------------------------------------------------------------
# Run-state snapshot (inputs to the approval algorithm)
# ---------------------------------------------------------------------------


@dataclass
class RunStateSnapshot:
    """Snapshot of run state features relevant to batch decisions."""

    generation: int
    total_generations: int
    meaningful_branch_count: int = 1
    recent_family_diversity: int = 1
    recent_extra_slot_yield: float = 0.0
    recent_duplicate_rate: float = 0.0
    recent_collapse_rate: float = 0.0
    stagnation_counter: int = 0
    recent_frontier_improvements: int = 0
    recent_best_improved: bool = False


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------


class BatchBudgetController:
    """Tracks slot budget and computes batch approvals.

    In Phase 3 (full adaptive), the engine can both shrink and expand
    the batch.  Expansion above nominal requires passing an explicit
    gate in the engine (see ``_gate_expansion``).
    """

    def __init__(
        self,
        total_generations: int,
        initial_batch_size: int = 1,
        config: BatchBudgetConfig | None = None,
    ) -> None:
        self.config = config or BatchBudgetConfig()
        self._total_generations = total_generations
        self._initial_batch_size = initial_batch_size
        self._total_slot_budget = total_generations * initial_batch_size
        self._slots_spent = 0
        self._history: list[GenerationYield] = []
        self._decisions: list[BatchDecision] = []

    # -- Budget queries -----------------------------------------------------

    @property
    def slots_remaining(self) -> int:
        return max(0, self._total_slot_budget - self._slots_spent)

    @property
    def generations_remaining(self, current_gen: int | None = None) -> int:
        last_gen = self._history[-1].generation if self._history else 0
        return max(1, self._total_generations - last_gen)

    @property
    def fair_share(self) -> float:
        """Neutral-pace slots per remaining generation."""
        return self.slots_remaining / self.generations_remaining

    @property
    def pace_ratio(self) -> float:
        """How far ahead/behind budget pace we are (>1 = overspent)."""
        if not self._history:
            return 1.0
        current_gen = self._history[-1].generation
        expected = current_gen * self._initial_batch_size
        return self._slots_spent / expected if expected > 0 else 1.0

    # -- Yield tracking -----------------------------------------------------

    def record_yield(self, gen_yield: GenerationYield) -> None:
        """Record what a generation's batch actually produced."""
        self._slots_spent += gen_yield.batch_size
        self._history.append(gen_yield)

    # -- Run-state feature extraction ---------------------------------------

    def _recent_yields(self) -> list[GenerationYield]:
        w = self.config.lookback_window
        return self._history[-w:] if self._history else []

    def _compute_recent_stats(self) -> dict[str, float]:
        recent = self._recent_yields()
        if not recent:
            return {
                "extra_slot_yield": 0.0,
                "duplicate_rate": 0.0,
                "collapse_rate": 0.0,
                "frontier_improvements": 0,
                "any_best_improved": False,
            }

        total_slots = sum(y.batch_size for y in recent)
        total_extra = sum(max(0, y.batch_size - 1) for y in recent)
        total_extra_successes = sum(max(0, y.successes - 1) for y in recent)
        total_duplicates = sum(y.duplicate_count for y in recent)
        total_collapsed = sum(y.collapsed_count for y in recent)
        total_frontier = sum(y.frontier_improvements for y in recent)

        return {
            "extra_slot_yield": (total_extra_successes / total_extra if total_extra > 0 else 0.0),
            "duplicate_rate": (total_duplicates / total_slots if total_slots > 0 else 0.0),
            "collapse_rate": (total_collapsed / total_slots if total_slots > 0 else 0.0),
            "frontier_improvements": total_frontier,
            "any_best_improved": any(y.best_improved for y in recent),
        }

    # -- Marginal value / cost heuristics -----------------------------------

    def _tier_multiplier(self, slot_index: int) -> float:
        """Cost multiplier for slot k (1-indexed)."""
        for i, boundary in enumerate(self.config.tier_boundaries):
            if slot_index < boundary:
                return self.config.tier_multipliers[i]
        return self.config.tier_multipliers[-1]

    def _marginal_value(self, slot_index: int, snapshot: RunStateSnapshot) -> float:
        """Estimate marginal value of adding slot k to the batch.

        Returns a value in [0, 1] where higher = more valuable.
        """
        v = 0.0

        # Branch diversity: more meaningful branches → higher value of exploration
        branch_bonus = min(1.0, snapshot.meaningful_branch_count / 4)
        v += 0.25 * branch_bonus

        # Family diversity: recent improvements from multiple families
        family_bonus = min(1.0, snapshot.recent_family_diversity / 3)
        v += 0.20 * family_bonus

        # Recent extra-slot yield: historical evidence that extra slots help
        v += 0.25 * snapshot.recent_extra_slot_yield

        # Stagnation: more stagnation → more value from widening
        stag_bonus = min(1.0, snapshot.stagnation_counter / 8)
        v += 0.15 * stag_bonus

        # Recent frontier activity: recent improvements → worth exploring more
        if snapshot.recent_frontier_improvements > 0:
            v += 0.10
        if snapshot.recent_best_improved:
            v += 0.05

        # Diminishing returns for later slots
        diminish = 1.0 / math.sqrt(slot_index)
        v *= diminish

        # Penalty for high duplicate/collapse rates
        dup_penalty = 1.0 - 0.5 * snapshot.recent_duplicate_rate
        collapse_penalty = 1.0 - 0.5 * snapshot.recent_collapse_rate
        v *= dup_penalty * collapse_penalty

        return max(0.0, min(1.0, v))

    def _marginal_cost(self, slot_index: int) -> float:
        """Estimate marginal cost of slot k given remaining budget.

        Returns a value in [0, 1] where higher = more expensive.
        """
        # Base cost from tier
        tier = self._tier_multiplier(slot_index)

        # Budget pressure: higher when few slots remain
        gens_left = self.generations_remaining
        budget_pressure = 1.0 - (self.slots_remaining / (gens_left * self.config.max_batch_size))
        budget_pressure = max(0.0, min(1.0, budget_pressure))

        # Pace penalty: overspending → higher cost
        pace = self.pace_ratio
        pace_penalty = max(0.0, pace - 1.0)  # 0 when on/under pace

        cost = 0.3 * (tier / self.config.tier_multipliers[-1])
        cost += 0.4 * budget_pressure
        cost += 0.3 * pace_penalty

        return max(0.0, min(1.0, cost))

    # -- Shadow approval algorithm ------------------------------------------

    def compute_approval(
        self,
        requested: int,
        actual_batch: int,
        snapshot: RunStateSnapshot,
    ) -> BatchDecision:
        """Compute batch approval.

        Parameters
        ----------
        requested
            What the controller estimates should be requested.
        actual_batch
            The nominal batch size (before shrink-only enforcement).
        snapshot
            Current run-state features.
        """
        max_possible = min(
            requested,
            self.config.max_batch_size,
            self.slots_remaining,
        )

        approved = 1  # Always approve at least 1
        trace: list[dict[str, float]] = []

        for k in range(2, max_possible + 1):
            mv = self._marginal_value(k, snapshot)
            mc = self._marginal_cost(k)
            entry = {"slot": k, "value": round(mv, 4), "cost": round(mc, 4)}

            if mv > mc:
                approved = k
                entry["approved"] = True
            else:
                entry["approved"] = False
                trace.append(entry)
                break  # Stop at first rejection

            trace.append(entry)

        decision = BatchDecision(
            requested=requested,
            approved=approved,
            actual=actual_batch,
            fair_share=self.fair_share,
            pace_ratio=self.pace_ratio,
            slots_remaining=self.slots_remaining,
            marginal_trace=trace,
        )
        self._decisions.append(decision)
        return decision

    def compute_heuristic_request(self, snapshot: RunStateSnapshot) -> int:
        """Engine-side heuristic batch request (Phase 1, no mutator input).

        Uses run-state features to estimate a reasonable batch request.
        """
        base = self._initial_batch_size

        # Widen when stagnating and branches are diverse
        if snapshot.stagnation_counter >= 5 and snapshot.meaningful_branch_count >= 2:
            base = min(base + 2, self.config.max_batch_size)

        # Widen when recent extra slots had high yield
        if snapshot.recent_extra_slot_yield > 0.3:
            base = min(base + 1, self.config.max_batch_size)

        # Shrink when duplicates or collapses dominate
        if snapshot.recent_duplicate_rate > 0.4 or snapshot.recent_collapse_rate > 0.4:
            base = max(1, base - 1)

        # Shrink when only one branch is productive
        if snapshot.meaningful_branch_count <= 1 and snapshot.stagnation_counter < 3:
            base = max(1, min(base, 2))

        # Budget cap
        base = min(base, self.slots_remaining)

        return max(1, base)

    # -- Persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_generations": self._total_generations,
            "initial_batch_size": self._initial_batch_size,
            "total_slot_budget": self._total_slot_budget,
            "slots_spent": self._slots_spent,
            "history": [y.to_dict() for y in self._history],
            "decisions": [d.to_dict() for d in self._decisions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchBudgetController:
        ctrl = cls(
            total_generations=d["total_generations"],
            initial_batch_size=d["initial_batch_size"],
        )
        ctrl._total_slot_budget = d["total_slot_budget"]
        ctrl._slots_spent = d["slots_spent"]
        ctrl._history = [GenerationYield.from_dict(y) for y in d.get("history", [])]
        ctrl._decisions = []  # Decisions are not needed for resume
        return ctrl
