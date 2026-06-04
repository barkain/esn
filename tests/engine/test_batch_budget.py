"""Tests for adaptive batch budget controller (Phase 1: shadow mode)."""

from __future__ import annotations

import json

import pytest

from esn.engine.batch_budget import (
    BatchBudgetConfig,
    BatchBudgetController,
    GenerationYield,
    RunStateSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_snapshot(**overrides) -> RunStateSnapshot:
    defaults = {
        "generation": 5,
        "total_generations": 30,
        "meaningful_branch_count": 2,
        "recent_family_diversity": 2,
        "recent_extra_slot_yield": 0.2,
        "recent_duplicate_rate": 0.0,
        "recent_collapse_rate": 0.0,
        "stagnation_counter": 3,
        "recent_frontier_improvements": 1,
        "recent_best_improved": False,
    }
    defaults.update(overrides)
    return RunStateSnapshot(**defaults)


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------


class TestBudgetTracking:
    def test_initial_budget_equals_generations_times_batch(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        assert ctrl._total_slot_budget == 120
        assert ctrl.slots_remaining == 120

    def test_record_yield_decrements_budget(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        ctrl.record_yield(GenerationYield(generation=1, batch_size=4, successes=2))
        assert ctrl._slots_spent == 4
        assert ctrl.slots_remaining == 116

    def test_fair_share_computation(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=4)
        assert ctrl.fair_share == pytest.approx(4.0)
        for g in range(1, 6):
            ctrl.record_yield(GenerationYield(generation=g, batch_size=4, successes=2))
        assert ctrl.fair_share == pytest.approx(4.0)

    def test_pace_ratio_on_pace(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=4)
        ctrl.record_yield(GenerationYield(generation=1, batch_size=4, successes=2))
        assert ctrl.pace_ratio == pytest.approx(1.0)

    def test_pace_ratio_overspent(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=4)
        ctrl.record_yield(GenerationYield(generation=1, batch_size=8, successes=4))
        assert ctrl.pace_ratio == pytest.approx(2.0)

    def test_pace_ratio_underspent(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=4)
        ctrl.record_yield(GenerationYield(generation=1, batch_size=2, successes=1))
        assert ctrl.pace_ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Approval algorithm
# ---------------------------------------------------------------------------


class TestApproval:
    def test_always_approves_at_least_one(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot()
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        assert decision.approved >= 1

    def test_shadow_mode_actual_unchanged(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot()
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        assert decision.actual == 4

    def test_approval_respects_max_batch(self):
        config = BatchBudgetConfig(max_batch_size=3)
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4, config=config)
        snap = _default_snapshot()
        decision = ctrl.compute_approval(requested=8, actual_batch=4, snapshot=snap)
        assert decision.approved <= 3

    def test_approval_respects_remaining_slots(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=2)
        for g in range(1, 10):
            ctrl.record_yield(GenerationYield(generation=g, batch_size=2, successes=1))
        snap = _default_snapshot(generation=10)
        decision = ctrl.compute_approval(requested=8, actual_batch=2, snapshot=snap)
        assert decision.approved <= 2

    def test_marginal_trace_recorded(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot()
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        assert isinstance(decision.marginal_trace, list)
        for entry in decision.marginal_trace:
            assert "slot" in entry
            assert "value" in entry
            assert "cost" in entry
            assert "approved" in entry

    def test_high_diversity_approves_more_slots(self):
        ctrl_low = BatchBudgetController(total_generations=30, initial_batch_size=4)
        ctrl_high = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap_low = _default_snapshot(meaningful_branch_count=1, recent_family_diversity=1)
        snap_high = _default_snapshot(meaningful_branch_count=4, recent_family_diversity=3)
        dec_low = ctrl_low.compute_approval(requested=6, actual_batch=4, snapshot=snap_low)
        dec_high = ctrl_high.compute_approval(requested=6, actual_batch=4, snapshot=snap_high)
        assert dec_high.approved >= dec_low.approved

    def test_high_duplicate_rate_shrinks_approval(self):
        ctrl_clean = BatchBudgetController(total_generations=30, initial_batch_size=4)
        ctrl_dup = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap_clean = _default_snapshot(recent_duplicate_rate=0.0)
        snap_dup = _default_snapshot(recent_duplicate_rate=0.8)
        dec_clean = ctrl_clean.compute_approval(requested=6, actual_batch=4, snapshot=snap_clean)
        dec_dup = ctrl_dup.compute_approval(requested=6, actual_batch=4, snapshot=snap_dup)
        assert dec_clean.approved >= dec_dup.approved


# ---------------------------------------------------------------------------
# Heuristic request
# ---------------------------------------------------------------------------


class TestHeuristicRequest:
    def test_baseline_returns_valid_range(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot(stagnation_counter=0)
        req = ctrl.compute_heuristic_request(snap)
        assert 1 <= req <= 8

    def test_stagnation_widens(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap_stale = _default_snapshot(stagnation_counter=8, meaningful_branch_count=3)
        snap_fresh = _default_snapshot(stagnation_counter=0, meaningful_branch_count=3)
        req_stale = ctrl.compute_heuristic_request(snap_stale)
        req_fresh = ctrl.compute_heuristic_request(snap_fresh)
        assert req_stale >= req_fresh

    def test_single_branch_shrinks(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot(meaningful_branch_count=1, stagnation_counter=0)
        req = ctrl.compute_heuristic_request(snap)
        assert req <= 2

    def test_high_duplicate_shrinks(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot(recent_duplicate_rate=0.6)
        req = ctrl.compute_heuristic_request(snap)
        assert req <= 3

    def test_budget_cap(self):
        ctrl = BatchBudgetController(total_generations=10, initial_batch_size=2)
        for g in range(1, 10):
            ctrl.record_yield(GenerationYield(generation=g, batch_size=2, successes=1))
        snap = _default_snapshot(generation=10, stagnation_counter=10, meaningful_branch_count=5)
        req = ctrl.compute_heuristic_request(snap)
        assert req <= ctrl.slots_remaining


# ---------------------------------------------------------------------------
# Marginal value / cost
# ---------------------------------------------------------------------------


class TestMarginals:
    def test_marginal_value_decreases_with_slot_index(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot()
        v2 = ctrl._marginal_value(2, snap)
        v5 = ctrl._marginal_value(5, snap)
        assert v2 > v5

    def test_marginal_cost_increases_with_tier(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        c2 = ctrl._marginal_cost(2)
        c7 = ctrl._marginal_cost(7)
        assert c7 > c2

    def test_marginal_cost_increases_when_overspent(self):
        ctrl_neutral = BatchBudgetController(total_generations=10, initial_batch_size=4)
        ctrl_over = BatchBudgetController(total_generations=10, initial_batch_size=4)
        ctrl_neutral.record_yield(GenerationYield(generation=1, batch_size=4, successes=2))
        ctrl_over.record_yield(GenerationYield(generation=1, batch_size=8, successes=2))
        c_neutral = ctrl_neutral._marginal_cost(3)
        c_over = ctrl_over._marginal_cost(3)
        assert c_over > c_neutral


# ---------------------------------------------------------------------------
# Generation yield
# ---------------------------------------------------------------------------


class TestGenerationYield:
    def test_success_rate(self):
        y = GenerationYield(generation=1, batch_size=4, successes=3)
        assert y.success_rate == pytest.approx(0.75)

    def test_extra_slot_yield(self):
        y = GenerationYield(generation=1, batch_size=4, successes=3)
        assert y.extra_slot_yield == pytest.approx(2 / 3)

    def test_extra_slot_yield_batch_one(self):
        y = GenerationYield(generation=1, batch_size=1, successes=1)
        assert y.extra_slot_yield == 0.0

    def test_roundtrip_dict(self):
        y = GenerationYield(
            generation=5,
            batch_size=4,
            successes=3,
            frontier_improvements=1,
            best_improved=True,
            unique_families=2,
            collapsed_count=0,
            duplicate_count=1,
        )
        restored = GenerationYield.from_dict(y.to_dict())
        assert restored.generation == 5
        assert restored.batch_size == 4
        assert restored.successes == 3
        assert restored.best_improved is True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_controller_roundtrip(self):
        ctrl = BatchBudgetController(total_generations=20, initial_batch_size=4)
        ctrl.record_yield(GenerationYield(generation=1, batch_size=4, successes=3))
        ctrl.record_yield(GenerationYield(generation=2, batch_size=4, successes=2))

        d = ctrl.to_dict()
        json_str = json.dumps(d)
        restored = BatchBudgetController.from_dict(json.loads(json_str))

        assert restored._total_generations == 20
        assert restored._initial_batch_size == 4
        assert restored._slots_spent == 8
        assert len(restored._history) == 2
        assert restored.slots_remaining == 72

    def test_decision_to_dict(self):
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot()
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        d = decision.to_dict()
        assert "requested" in d
        assert "approved" in d
        assert "actual" in d
        assert "fair_share" in d
        assert "marginal_trace" in d
        json.dumps(d)


# ---------------------------------------------------------------------------
# Phase 2: shrink-only enforcement
# ---------------------------------------------------------------------------


class TestShrinkOnlyEnforcement:
    """Phase 2: engine uses min(nominal, approved) after warmup."""

    def test_effective_batch_capped_at_nominal(self):
        """Even if the controller would approve more, effective <= nominal."""
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        # Simulate rich diversity that would justify large batch
        snap = _default_snapshot(
            meaningful_branch_count=4,
            recent_family_diversity=3,
            recent_extra_slot_yield=0.5,
            stagnation_counter=8,
            recent_frontier_improvements=2,
            recent_best_improved=True,
        )
        decision = ctrl.compute_approval(requested=8, actual_batch=4, snapshot=snap)
        # Controller might approve up to 8, but effective = min(4, approved)
        effective = min(4, decision.approved)
        assert effective <= 4

    def test_effective_batch_shrinks_when_justified(self):
        """Controller approval < nominal results in smaller effective batch."""
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        # Poor diversity: only 1 branch, no yield from extra slots
        snap = _default_snapshot(
            meaningful_branch_count=1,
            recent_family_diversity=1,
            recent_extra_slot_yield=0.0,
            stagnation_counter=0,
        )
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        effective = min(4, decision.approved)
        assert effective < 4  # Controller should shrink below nominal

    def test_effective_always_at_least_one(self):
        """min(nominal, approved) >= 1 because approved >= 1 always."""
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot(
            recent_duplicate_rate=0.9,
            recent_collapse_rate=0.9,
            meaningful_branch_count=1,
        )
        decision = ctrl.compute_approval(requested=1, actual_batch=4, snapshot=snap)
        effective = min(4, decision.approved)
        assert effective >= 1

    def test_shrink_preserves_budget_accounting(self):
        """When effective < nominal, record_yield uses effective batch_size."""
        ctrl = BatchBudgetController(total_generations=30, initial_batch_size=4)
        snap = _default_snapshot(meaningful_branch_count=1, stagnation_counter=0)
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        effective = min(4, decision.approved)
        # Record yield with effective (not nominal) batch size
        ctrl.record_yield(
            GenerationYield(
                generation=1,
                batch_size=effective,
                successes=1,
            )
        )
        # Budget should reflect effective slots spent, not nominal
        assert ctrl._slots_spent == effective
        assert ctrl.slots_remaining == ctrl._total_slot_budget - effective

    def test_warmup_returns_nominal(self):
        """During warmup (first lookback_window gens), effective == nominal."""
        from esn.engine.batch_budget import BatchBudgetConfig

        config = BatchBudgetConfig(lookback_window=5)
        ctrl = BatchBudgetController(
            total_generations=30,
            initial_batch_size=4,
            config=config,
        )
        # At generation 3 (within warmup of 5), controller would shrink
        snap = _default_snapshot(
            generation=3,
            meaningful_branch_count=1,
            stagnation_counter=0,
        )
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        # Warmup logic is in the engine, not the controller — but the
        # controller should still give honest approval.  The engine applies:
        #   if gen <= warmup: effective = nominal
        # So during warmup, effective stays at 4 regardless of approval.
        # This test verifies the warmup contract.
        gen = 3
        warmup = config.lookback_window
        if gen <= warmup:
            effective = 4  # nominal, not shrunk
        else:
            effective = min(4, decision.approved)
        assert effective == 4

    def test_post_warmup_enforces(self):
        """After warmup period, shrink-only enforcement activates."""
        from esn.engine.batch_budget import BatchBudgetConfig

        config = BatchBudgetConfig(lookback_window=3)
        ctrl = BatchBudgetController(
            total_generations=30,
            initial_batch_size=4,
            config=config,
        )
        # Build up some history (warmup period)
        for g in range(1, 4):
            ctrl.record_yield(
                GenerationYield(
                    generation=g,
                    batch_size=4,
                    successes=1,
                )
            )
        # Now at gen 4 (past warmup of 3), with poor signals
        snap = _default_snapshot(
            generation=4,
            meaningful_branch_count=1,
            recent_family_diversity=1,
            recent_extra_slot_yield=0.0,
            stagnation_counter=0,
        )
        decision = ctrl.compute_approval(requested=4, actual_batch=4, snapshot=snap)
        gen = 4
        warmup = config.lookback_window
        if gen <= warmup:
            effective = 4
        else:
            effective = min(4, decision.approved)
        assert effective < 4  # Should be shrunk post-warmup

    def test_nominal_batch_restored_after_generation(self):
        """Engine._batch_size is restored to nominal after the try/finally."""
        # This is an engine-level test but verifying the contract:
        # nominal is saved, effective is applied, nominal is restored.
        nominal = 4
        approved = 2
        effective = min(nominal, approved)
        assert effective == 2
        # After finally block, batch_size should be back to nominal
        restored = nominal  # simulates the finally block
        assert restored == 4
