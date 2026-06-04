"""Tests for ESN engine batched generation (batch_size > 1)."""

from __future__ import annotations

import pytest

from esn.core.enums import SearchMode
from esn.core.models import EvaluationResult
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine, _CandidateOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_CODE = "def solve():\n    return [1, 2, 3]\n"
BETTER_CODE = "def solve():\n    return [10, 20, 30]\n"
FAILING_CODE = "def solve():\n    raise RuntimeError('boom')\n"


def _sum_evaluator(artifact):
    if artifact is None:
        return EvaluationResult(score=0.0, success=False)
    try:
        total = float(sum(artifact))
        return EvaluationResult(score=total, success=True)
    except Exception:
        return EvaluationResult(score=0.0, success=False)


def _make_domain(initial_code: str = SIMPLE_CODE) -> DomainSpec:
    return DomainSpec(
        name="test",
        description="batch test domain",
        initial_code=initial_code,
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=_sum_evaluator,
        allowed_imports=frozenset({"math"}),
    )


def _make_engine(batch_size: int = 1, **kwargs) -> ESNEngine:
    domain = kwargs.pop("domain", _make_domain())
    return ESNEngine(domain=domain, batch_size=batch_size, **kwargs)


# ---------------------------------------------------------------------------
# 1. batch_size=1 remains behaviorally identical
# ---------------------------------------------------------------------------


class TestBatchSizeOneIdentical:
    """batch_size=1 must behave identically to the original single path."""

    def test_single_gen_happy_path(self):
        engine = _make_engine(batch_size=1)
        record = engine.run_generation()
        assert record.success is True
        assert record.score == 6.0
        assert record.generation == 1

    def test_generation_increments(self):
        engine = _make_engine(batch_size=1)
        for i in range(3):
            record = engine.run_generation()
            assert record.generation == i + 1

    def test_best_score_tracked(self):
        engine = _make_engine(batch_size=1)
        engine.run_generation()
        assert engine._best_score == 6.0

    def test_archives_populated(self):
        engine = _make_engine(batch_size=1)
        engine.run_generation()
        assert engine.elite_archive.size >= 1

    def test_credit_recorded(self):
        engine = _make_engine(batch_size=1)
        engine.run_generation()
        # Without mutator, the identity path runs refine style
        total = sum(
            engine.credit_model.get_stats(s).attempts
            for s in ["refine", "explore", "repair", "radical"]
        )
        assert total >= 1


# ---------------------------------------------------------------------------
# 2. Batch mode processes multiple outcomes without corrupting shared state
# ---------------------------------------------------------------------------


class TestBatchMultipleOutcomes:
    """Batch mode with batch_size > 1 produces correct multi-candidate results."""

    def test_batch_returns_list(self):
        engine = _make_engine(batch_size=4)
        records = engine.run_batch_generation()
        # Without a mutator, all candidates use identity mutation on best_code
        # so all should succeed with the same score
        assert len(records) == 4
        for r in records:
            assert r.success is True
            assert r.score == 6.0
            assert r.generation == 1

    def test_run_generation_returns_best(self):
        """run_generation() wrapper returns the best successful candidate."""
        engine = _make_engine(batch_size=4)
        record = engine.run_generation()
        assert record.success is True
        assert record.score == 6.0

    def test_generation_counter_increments_once_per_batch(self):
        """A batch of k candidates is ONE generation, not k generations."""
        engine = _make_engine(batch_size=4)
        engine.run_generation()
        assert engine.generation == 1
        engine.run_generation()
        assert engine.generation == 2

    def test_state_consistent_after_batch(self):
        engine = _make_engine(batch_size=4)
        engine.run_generation()
        s = engine.state
        assert s.generation == 1
        assert s.best_score == 6.0
        assert s.elite_size >= 1
        # Recent scores should have entries for all batch candidates
        assert len(s.recent_scores) >= 4

    def test_program_store_has_all_candidates(self):
        engine = _make_engine(batch_size=4)
        records = engine.run_batch_generation()
        for r in records:
            if r.success:
                assert r.id in engine._program_store


# ---------------------------------------------------------------------------
# 3. Parent/style portfolio selection is diverse and valid
# ---------------------------------------------------------------------------


class TestPortfolioSelection:
    """Parent and style portfolio methods produce valid, diverse selections."""

    def test_parent_portfolio_always_has_best(self):
        engine = _make_engine(batch_size=4)
        engine._evaluate_seed_if_needed()
        parents = engine._select_parent_portfolio(SearchMode.EXPLOIT)
        assert engine._best_code in parents

    def test_parent_portfolio_deduplicates(self):
        engine = _make_engine(batch_size=4)
        engine._evaluate_seed_if_needed()
        parents = engine._select_parent_portfolio(SearchMode.EXPLOIT)
        # All parents should have unique content hashes
        import hashlib

        hashes = [hashlib.sha256(p.encode()).hexdigest()[:16] for p in parents]
        assert len(hashes) == len(set(hashes))

    def test_parent_portfolio_nonempty(self):
        engine = _make_engine(batch_size=8)
        parents = engine._select_parent_portfolio(SearchMode.EXPLORE)
        assert len(parents) >= 1

    @pytest.mark.skip(
        reason="_select_style_portfolio is legacy dead code; "
        "batch_size>1 now uses BatchSlotScorer, batch_size=1 uses _select_style"
    )
    def test_style_portfolio_has_refine_and_explore(self):
        engine = _make_engine(batch_size=4)
        # Run enough gens to get past forced exploration
        for _ in range(8):
            engine.run_generation()
        styles = engine._select_style_portfolio(SearchMode.EXPLOIT, 4)
        assert "refine" in styles
        assert "explore" in styles

    def test_style_portfolio_correct_size(self):
        engine = _make_engine(batch_size=6)
        styles = engine._select_style_portfolio(SearchMode.EXPLOIT, 6)
        assert len(styles) == 6

    def test_style_portfolio_forced_exploration(self):
        """Early gens should get forced exploration styles."""
        engine = _make_engine(batch_size=4)
        styles = engine._select_style_portfolio(SearchMode.EXPLOIT, 4)
        # Should include untried core styles
        assert len(styles) == 4


# ---------------------------------------------------------------------------
# 4. Failures in one candidate do not poison other candidates
# ---------------------------------------------------------------------------


class TestBatchFailureIsolation:
    """A failing candidate in the batch must not corrupt successful ones."""

    def test_mixed_success_and_failure(self):
        """Simulate batch where some candidates fail at eval stage."""
        engine = _make_engine(batch_size=1)
        engine._evaluate_seed_if_needed()
        engine.generation += 1
        engine.state.generation = engine.generation

        mode = SearchMode.EXPLOIT
        # Create a successful outcome
        ok = _CandidateOutcome(
            slot=0,
            style="refine",
            mode=mode,
            parent_code=SIMPLE_CODE,
            success=True,
            score=6.0,
            raw_score=6.0,
            new_code=SIMPLE_CODE,
            code_hash="abc123",
            family="test",
            family_confidence="high",
            solve_summary="test",
        )
        ok.eval_result = EvaluationResult(score=6.0, success=True)

        # Create a failed outcome
        fail = _CandidateOutcome(
            slot=1,
            style="explore",
            mode=mode,
            parent_code=SIMPLE_CODE,
            success=False,
            score=0.0,
            raw_score=0.0,
            new_code=FAILING_CODE,
            failure_stage="compile",
            errors=["RuntimeError: boom"],
            family="test_fail",
        )

        # Process both outcomes sequentially (Phase 3)
        ok_record = engine._process_outcome(ok)
        fail_record = engine._process_outcome(fail)

        assert ok_record.success is True
        assert ok_record.score == 6.0
        assert fail_record.success is False
        # Failure records must persist family/family_confidence from the
        # outcome for downstream analytics (archive_families, mutator
        # failure-reason summaries). Prior to the fix these defaulted
        # back to CandidateRecord's defaults ("", "none") instead of
        # reflecting the outcome, making failures invisible to family-
        # aware reporting.
        assert fail_record.family == "test_fail"
        # _CandidateOutcome.family_confidence default is "" — the fix
        # must propagate that literal, not leave CandidateRecord's
        # default of "none" in place.
        assert fail_record.family_confidence == ""

    def test_all_fail_increments_consecutive_failures(self):
        engine = _make_engine(batch_size=1)
        engine._evaluate_seed_if_needed()
        engine.generation += 1
        engine.state.generation = engine.generation

        mode = SearchMode.EXPLOIT
        fail1 = _CandidateOutcome(
            slot=0,
            style="refine",
            mode=mode,
            parent_code=SIMPLE_CODE,
            failure_stage="mutation",
            errors=["LLM error"],
        )
        fail2 = _CandidateOutcome(
            slot=1,
            style="explore",
            mode=mode,
            parent_code=SIMPLE_CODE,
            failure_stage="compile",
            errors=["syntax error"],
        )

        engine._process_outcome(fail1)
        engine._process_outcome(fail2)
        engine._finalize_batch([fail1, fail2], any_success=False, best_outcome=None)

        assert engine._consecutive_failures == 1

    def test_any_success_resets_consecutive_failures(self):
        engine = _make_engine(batch_size=1)
        engine._consecutive_failures = 3  # simulate prior failures
        engine._evaluate_seed_if_needed()
        engine.generation += 1
        engine.state.generation = engine.generation

        mode = SearchMode.EXPLOIT
        ok = _CandidateOutcome(
            slot=0,
            style="refine",
            mode=mode,
            parent_code=SIMPLE_CODE,
            success=True,
            score=6.0,
            raw_score=6.0,
            new_code=SIMPLE_CODE,
            code_hash="abc123",
            family="test",
            family_confidence="high",
        )
        ok.eval_result = EvaluationResult(score=6.0, success=True)

        engine._process_outcome(ok)
        engine._finalize_batch([ok], any_success=True, best_outcome=ok)

        assert engine._consecutive_failures == 0

    def test_timeout_outcome_persists_source_in_program_store(self):
        """A timed-out candidate's source MUST be persisted in program_store.

        Regression test: before the fix, only successful candidates were
        written to ``_program_store``. Failure-path candidates (including
        compile-stage timeouts, where ``failure_stage="compile"`` and
        ``compile_metadata["stage"]="timeout"``) had their source dropped,
        which meant ``programs.json`` was success-only and the benchmark
        audit-log embed silently omitted the triple-backtick code block
        for timed-out programs. Fixing ``_program_store`` fixes both
        downstream consumers: programs.json round-trips the source, and
        the audit-log writer's ``if code:`` check now sees non-empty code
        and includes the block.
        """
        engine = _make_engine(batch_size=1)
        engine._evaluate_seed_if_needed()
        engine.generation += 1
        engine.state.generation = engine.generation

        timeout_code = "def solve():\n    while True:\n        pass\n"
        mode = SearchMode.EXPLOIT
        # Mirrors what the compilers (compiler.py / uv_compiler.py /
        # stdio_compiler.py) produce on subprocess.TimeoutExpired:
        # failure_stage="compile", errors[0] starts with "Timeout:".
        timeout_outcome = _CandidateOutcome(
            slot=0,
            style="refine",
            mode=mode,
            parent_code=SIMPLE_CODE,
            new_code=timeout_code,
            failure_stage="compile",
            errors=["Timeout: execution exceeded 60s"],
            family="test_timeout",
        )

        record = engine._process_outcome(timeout_outcome)

        assert record.success is False
        assert record.id in engine._program_store
        assert engine._program_store[record.id] == timeout_code
        # compile_metadata carries the stage; downstream consumers use it
        # to distinguish timeout from other compile-stage failures.
        assert record.compile_metadata is not None
        assert record.compile_metadata.get("stage") == "compile"
        # Errors must include the Timeout marker so audit-log readers can
        # classify the failure without re-running the program.
        assert any("Timeout" in e for e in record.compile_metadata.get("errors", []))

    def test_non_timeout_failure_also_persists_source(self):
        """Symmetric case: mutation/validation failures with code also persist."""
        engine = _make_engine(batch_size=1)
        engine._evaluate_seed_if_needed()
        engine.generation += 1
        engine.state.generation = engine.generation

        bad_code = "def solve():\n    return 1/0\n"
        fail = _CandidateOutcome(
            slot=0,
            style="explore",
            mode=SearchMode.EXPLOIT,
            parent_code=SIMPLE_CODE,
            new_code=bad_code,
            failure_stage="compile",
            errors=["ZeroDivisionError"],
            family="test",
        )
        record = engine._process_outcome(fail)
        assert record.id in engine._program_store
        assert engine._program_store[record.id] == bad_code


# ---------------------------------------------------------------------------
# 5. Archives/credit/knowledge update deterministically after batch
# ---------------------------------------------------------------------------


class TestBatchArchivingAndCredit:
    """Archives and credit model are updated correctly for each candidate."""

    def test_all_successful_candidates_archived(self):
        engine = _make_engine(batch_size=4)
        records = engine.run_batch_generation()
        assert records  # batch produced candidate records
        # Each successful candidate should be in elite or frontier
        assert engine.elite_archive.size >= 1

    def test_credit_recorded_for_each_candidate(self):
        engine = _make_engine(batch_size=4)
        engine.run_batch_generation()
        # Total credit attempts should equal batch size
        total = sum(
            engine.credit_model.get_stats(s).attempts
            for s in ["refine", "explore", "repair", "radical"]
        )
        assert total >= 4

    def test_recent_attempt_log_scaled(self):
        """Rolling attempt log should scale with batch size."""
        engine = _make_engine(batch_size=4)
        # Run several batches
        for _ in range(5):
            engine.run_batch_generation()
        # Log cap should be max(8, 2 * batch_size) = 8
        assert len(engine._recent_attempt_log) <= 8

    def test_recent_attempt_log_large_batch(self):
        engine = _make_engine(batch_size=8)
        for _ in range(5):
            engine.run_batch_generation()
        # Log cap should be max(8, 2 * 8) = 16
        assert len(engine._recent_attempt_log) <= 16

    def test_seen_hashes_populated(self):
        engine = _make_engine(batch_size=4)
        engine.run_batch_generation()
        # All successful candidates' hashes should be tracked
        assert len(engine._seen_hashes) >= 1


# ---------------------------------------------------------------------------
# 6. Best-score and consecutive-failure handling at batch level
# ---------------------------------------------------------------------------


class TestBatchBestScoreAndStagnation:
    """Best score updates and stagnation behave correctly across batches."""

    def test_best_score_updated_from_batch(self):
        engine = _make_engine(batch_size=4)
        engine.run_batch_generation()
        assert engine._best_score == 6.0

    def test_stagnation_increments_when_no_improvement(self):
        engine = _make_engine(batch_size=4)
        # First batch sets the score
        engine.run_batch_generation()
        initial_stagnation = engine.state.stagnation_counter
        # Second batch has same code, no improvement
        engine.run_batch_generation()
        assert engine.state.stagnation_counter > initial_stagnation

    def test_breakthrough_resets_stagnation(self):
        domain = _make_domain()
        engine = _make_engine(batch_size=1, domain=domain)
        # Run a few gens to build up stagnation
        for _ in range(5):
            engine.run_generation()
        assert engine.state.stagnation_counter >= 1

        # Simulate breakthrough by changing best_code to something much better
        engine._best_code = BETTER_CODE
        domain.initial_code = BETTER_CODE
        engine.run_generation()
        # Score 60.0 >> 6.0 * 1.005, so stagnation should reset
        assert engine._best_score == 60.0
        assert engine.state.stagnation_counter == 0

    def test_breakthrough_cooldown_set(self):
        domain = _make_domain()
        engine = _make_engine(batch_size=1, domain=domain)
        engine.run_generation()  # score = 6.0
        # Force a breakthrough
        engine._best_code = BETTER_CODE
        domain.initial_code = BETTER_CODE
        engine.run_generation()  # score = 60.0
        assert engine._breakthrough_cooldown > 0

    def test_consecutive_failure_recovery_in_batch(self):
        """After 2+ consecutive failures, batch should snap to best+refine."""
        engine = _make_engine(batch_size=4)
        engine._consecutive_failures = 3
        records = engine.run_batch_generation()
        # Should recover with refine style on best code
        assert any(r.success for r in records)
        # Consecutive failures should be reset
        assert engine._consecutive_failures == 0


# ---------------------------------------------------------------------------
# 7. Persistence round-trip with batch_size
# ---------------------------------------------------------------------------


class TestBatchPersistence:
    """batch_size is saved/loaded correctly."""

    def test_save_includes_batch_size(self, tmp_path):
        engine = _make_engine(batch_size=4)
        engine.run_batch_generation()
        engine.save_state(tmp_path / "state")
        import json

        v3_state = json.loads((tmp_path / "state" / "v3_state.json").read_text())
        assert v3_state["batch_size"] == 4

    def test_load_restores_batch_size(self, tmp_path):
        """On resume, checkpoint batch_size wins over constructor value.

        Regression: load_state used to silently drop the persisted
        batch_size, so resuming under a different CLI default silently
        widened the nominal batch.
        """
        # Save with batch_size=2
        saver = _make_engine(batch_size=2)
        saver.run_batch_generation()
        saver.save_state(tmp_path / "state")

        # Load into an engine constructed with a different batch_size
        loader = _make_engine(batch_size=8)
        loader.load_state(tmp_path / "state")
        assert loader._batch_size == 2


# ---------------------------------------------------------------------------
# 8. Phase 2: shrink-only enforcement at engine level
# ---------------------------------------------------------------------------


class TestPhase2ShrinkOnly:
    """Phase 2: engine applies min(nominal, approved) after warmup."""

    def test_warmup_preserves_full_batch(self):
        """During warmup period, full nominal batch is used."""
        engine = _make_engine(batch_size=4, total_generations=30)
        # Generation 1 is within warmup (default lookback_window=5)
        records = engine.run_batch_generation()
        assert len(records) == 4

    def test_batch_size_restored_after_generation(self):
        """_batch_size is restored to nominal after the generation."""
        engine = _make_engine(batch_size=4, total_generations=30)
        engine.run_batch_generation()
        assert engine._batch_size == 4

    def test_decision_actual_reflects_effective(self):
        """The decision dict's 'actual' field reflects the effective batch."""
        engine = _make_engine(batch_size=4, total_generations=30)
        engine.run_batch_generation()
        decision = engine._last_batch_decision
        # During warmup, actual should equal nominal (4)
        assert decision["actual"] == 4

    def test_multiple_generations_preserve_batch_size(self):
        """After multiple generations, nominal batch_size is always restored."""
        engine = _make_engine(batch_size=4, total_generations=30)
        for _ in range(6):
            engine.run_batch_generation()
        assert engine._batch_size == 4

    def test_yield_records_effective_batch(self):
        """GenerationYield.batch_size records effective (actual) candidates."""
        engine = _make_engine(batch_size=4, total_generations=30)
        engine.run_batch_generation()
        history = engine._batch_budget._history
        assert len(history) == 1
        # During warmup, batch_size should be 4 (full nominal)
        assert history[0].batch_size == 4

    def test_post_warmup_shrinks_batch(self):
        """After warmup, shrink-only enforcement actually reduces slot count."""
        from esn.engine.batch_budget import BatchBudgetConfig

        config = BatchBudgetConfig(lookback_window=2)
        engine = _make_engine(batch_size=4, total_generations=30)
        engine._batch_budget.config = config

        # Run 2 warmup generations (lookback_window=2)
        for _ in range(2):
            records = engine.run_batch_generation()
            assert len(records) == 4  # Full batch during warmup

        # Generation 3: past warmup, controller should shrink because
        # all candidates are identity-mutated duplicates with no frontier
        # improvements and only one branch.
        records = engine.run_batch_generation()
        assert len(records) < 4  # Shrink-only enforcement kicked in
        assert len(records) >= 1  # Always at least 1

        # Verify nominal batch_size is restored
        assert engine._batch_size == 4

        # Verify the decision dict's actual matches what ran
        assert engine._last_batch_decision["actual"] == len(records)

    def test_post_warmup_decision_persisted_correctly(self, tmp_path):
        """batch_budget.json decisions have correct actual after enforcement."""
        from esn.engine.batch_budget import BatchBudgetConfig
        import json

        config = BatchBudgetConfig(lookback_window=2)
        engine = _make_engine(batch_size=4, total_generations=30)
        engine._batch_budget.config = config

        # Run past warmup
        for _ in range(3):
            engine.run_batch_generation()

        # Save and check batch_budget.json
        engine.save_state(tmp_path / "state")
        budget_data = json.loads((tmp_path / "state" / "batch_budget.json").read_text())

        # The third decision (post-warmup) should have actual < 4
        decisions = budget_data["decisions"]
        assert len(decisions) == 3
        # Warmup decisions have actual == 4
        assert decisions[0]["actual"] == 4
        assert decisions[1]["actual"] == 4
        # Post-warmup decision has actual < 4 (shrink-only)
        assert decisions[2]["actual"] < 4


# ---------------------------------------------------------------------------
# 9. Phase 3: full adaptive enforcement (expansion gate)
# ---------------------------------------------------------------------------


class TestPhase3Expansion:
    """Phase 3: engine allows expansion above nominal when gates pass."""

    def test_expansion_blocked_single_branch(self):
        """Expansion blocked when only one meaningful branch."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=1,
            recent_family_diversity=3,
            recent_extra_slot_yield=0.5,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: single branch

    def test_expansion_blocked_single_family(self):
        """Expansion blocked when only one family producing results."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=1,
            recent_extra_slot_yield=0.5,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: single family

    def test_expansion_blocked_no_extra_yield(self):
        """Expansion blocked when extra slots have never helped."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=2,
            recent_extra_slot_yield=0.0,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: no evidence extra slots help

    def test_expansion_blocked_high_duplicates(self):
        """Expansion blocked when too many duplicates."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=2,
            recent_extra_slot_yield=0.5,
            recent_duplicate_rate=0.5,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: high duplicate rate

    def test_expansion_blocked_high_collapse(self):
        """Expansion blocked when too many collapses."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=2,
            recent_extra_slot_yield=0.5,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.5,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: high collapse rate

    def test_expansion_blocked_overspent(self):
        """Expansion blocked when already overspent."""
        from esn.engine.batch_budget import GenerationYield, RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=10)
        # Overspend: record 8 slots for gen 1 (pace_ratio = 2.0)
        engine._batch_budget.record_yield(GenerationYield(generation=1, batch_size=8, successes=4))
        snap = RunStateSnapshot(
            generation=10,
            total_generations=10,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=2,
            recent_extra_slot_yield=0.5,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 4  # blocked: pace_ratio > 1.2

    def test_expansion_allowed_when_all_gates_pass(self):
        """Expansion allowed when all gates are satisfied."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=3,
            recent_family_diversity=2,
            recent_extra_slot_yield=0.4,
            recent_duplicate_rate=0.1,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 6, snap)
        assert result == 6  # all gates pass: expand to approved

    def test_expansion_returns_approved_not_more(self):
        """Expansion never exceeds what the controller approved."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=10,
            total_generations=30,
            recent_best_improved=True,
            meaningful_branch_count=5,
            recent_family_diversity=4,
            recent_extra_slot_yield=0.8,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(4, 5, snap)
        assert result == 5  # exactly approved, not more

    def test_expansion_end_to_end(self):
        """Full integration: engine actually runs more candidates than nominal."""
        from unittest.mock import patch
        from esn.engine.batch_budget import BatchBudgetConfig, RunStateSnapshot

        config = BatchBudgetConfig(lookback_window=2)
        engine = _make_engine(batch_size=4, total_generations=30)
        engine._batch_budget.config = config

        # Run 2 warmup generations normally
        for _ in range(2):
            records = engine.run_batch_generation()
            assert len(records) == 4

        # Patch _build_batch_snapshot to return a favorable snapshot
        # that will make the controller approve 5 (> nominal 4)
        favorable_snap = RunStateSnapshot(
            generation=3,
            total_generations=30,
            meaningful_branch_count=4,
            recent_family_diversity=3,
            recent_extra_slot_yield=0.8,
            stagnation_counter=8,
            recent_frontier_improvements=2,
            recent_best_improved=True,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        with patch.object(engine, "_build_batch_snapshot", return_value=favorable_snap):
            records = engine.run_batch_generation()

        # The engine should have run MORE than nominal candidates
        assert len(records) > 4, f"Expected > 4 records, got {len(records)}"
        assert len(records) == 5  # controller approves 5 with this snapshot

        # Nominal batch_size must be restored
        assert engine._batch_size == 4

        # Decision dict should reflect the expanded effective batch
        assert engine._last_batch_decision["actual"] == 5
        assert engine._last_batch_decision["approved"] == 5

        # Yield history should record the actual 5 candidates
        last_yield = engine._batch_budget._history[-1]
        assert last_yield.batch_size == 5

    def test_expansion_persisted_correctly(self, tmp_path):
        """batch_budget.json records expanded actual after enforcement."""
        from unittest.mock import patch
        from esn.engine.batch_budget import BatchBudgetConfig, RunStateSnapshot
        import json

        config = BatchBudgetConfig(lookback_window=2)
        engine = _make_engine(batch_size=4, total_generations=30)
        engine._batch_budget.config = config

        # Warmup
        for _ in range(2):
            engine.run_batch_generation()

        # Expanded generation
        favorable_snap = RunStateSnapshot(
            generation=3,
            total_generations=30,
            meaningful_branch_count=4,
            recent_family_diversity=3,
            recent_extra_slot_yield=0.8,
            stagnation_counter=8,
            recent_frontier_improvements=2,
            recent_best_improved=True,
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        with patch.object(engine, "_build_batch_snapshot", return_value=favorable_snap):
            engine.run_batch_generation()

        # Save and verify batch_budget.json
        engine.save_state(tmp_path / "state")
        budget_data = json.loads((tmp_path / "state" / "batch_budget.json").read_text())
        decisions = budget_data["decisions"]
        assert len(decisions) == 3
        # Warmup: actual == 4
        assert decisions[0]["actual"] == 4
        assert decisions[1]["actual"] == 4
        # Expanded: actual == 5
        assert decisions[2]["actual"] == 5

    def test_expansion_blocked_no_momentum(self):
        """Gate 6: expansion blocked when recent_best_improved is False,
        even when all other gates pass."""
        from esn.engine.batch_budget import RunStateSnapshot

        engine = _make_engine(batch_size=4, total_generations=30)
        snap = RunStateSnapshot(
            generation=5,
            total_generations=30,
            meaningful_branch_count=4,
            recent_family_diversity=3,
            recent_extra_slot_yield=0.8,
            stagnation_counter=3,
            recent_frontier_improvements=2,
            recent_best_improved=False,  # the failing gate
            recent_duplicate_rate=0.0,
            recent_collapse_rate=0.0,
        )
        result = engine._gate_expansion(nominal=4, approved=6, snapshot=snap)
        assert result == 4  # nominal - momentum gate blocks expansion


# ---------------------------------------------------------------------------
# 10. Adaptive batching report formatter
# ---------------------------------------------------------------------------


class TestAdaptiveBatchingReport:
    """Pure formatter: renders an '## Adaptive Batching' markdown section."""

    def _make_controller(self):
        from esn.engine.batch_budget import (
            BatchBudgetController,
            BatchDecision,
            GenerationYield,
        )

        controller = BatchBudgetController(
            total_generations=10,
            initial_batch_size=4,
        )
        # Simulate 3 recorded generations: 4, 4, 2 effective candidates.
        controller.record_yield(GenerationYield(generation=1, batch_size=4, successes=3))
        controller.record_yield(GenerationYield(generation=2, batch_size=4, successes=2))
        controller.record_yield(GenerationYield(generation=3, batch_size=2, successes=1))
        # Synthetic decisions reflect real engine flow:
        # compute_approval() sets `actual` to the nominal batch, then the
        # engine mutates `actual` to the post-gating effective batch.
        # With nominal_batch_size=4:
        #   [0] shrink     — actual=2 < 4 (budget/pace shrank the batch)
        #   [1] no-change  — actual=4 == 4 (engine ran full nominal)
        #   [2] expand     — actual=5 > 4 (expansion gate approved +1)
        controller._decisions = [
            BatchDecision(
                requested=4,
                approved=2,
                actual=2,
                fair_share=2.0,
                pace_ratio=1.1,
                slots_remaining=30,
            ),
            BatchDecision(
                requested=4,
                approved=4,
                actual=4,
                fair_share=4.0,
                pace_ratio=1.0,
                slots_remaining=26,
            ),
            BatchDecision(
                requested=4,
                approved=5,
                actual=5,
                fair_share=4.0,
                pace_ratio=0.9,
                slots_remaining=21,
            ),
        ]
        return controller

    def test_renders_basic_section(self):
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = self._make_controller()
        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=False,
        )
        output = "\n".join(lines)

        assert "## Adaptive Batching" in output
        assert "- Nominal batch size: **4**" in output
        # total_slots = total_generations * initial_batch_size = 10 * 4 = 40
        # spent = 4 + 4 + 2 = 10; saved = 40 - 10 = 30
        assert "budgeted **40**" in output
        assert "spent **10**" in output
        assert "saved **30**" in output
        # Shrink: decision[0].actual=2 < nominal=4
        # Expand: decision[2].actual=5 > nominal=4
        # decision[1].actual=4 == nominal=4 (neither)
        assert "Shrink events: **1**" in output
        assert "(effective < nominal)" in output
        assert "Expand events: **1**" in output
        assert "(effective > nominal)" in output
        assert "### Effective-batch histogram" in output
        assert "| Size | Count |" in output
        # Histogram rows (ascending): size 2 -> 1, size 4 -> 2
        assert "| 2 | 1 |" in output
        assert "| 4 | 2 |" in output

    def test_resume_line_present_when_flag_true(self):
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = self._make_controller()
        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=True,
        )
        assert any(
            "Resume: restored batch size from checkpoint (CLI override ignored)" in line
            for line in lines
        )

    def test_resume_line_absent_when_flag_false(self):
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = self._make_controller()
        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=False,
        )
        assert not any("Resume: restored" in line for line in lines)

    def test_empty_history_renders_placeholder(self):
        from esn.engine.batch_budget import BatchBudgetController
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = BatchBudgetController(
            total_generations=10,
            initial_batch_size=4,
        )
        # No yields recorded and no decisions.
        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=False,
        )
        output = "\n".join(lines)
        assert "- No generations recorded yet." in output
        assert "### Effective-batch histogram" not in output

    def test_empty_decisions_zero_counts(self):
        from esn.engine.batch_budget import (
            BatchBudgetController,
            GenerationYield,
        )
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = BatchBudgetController(
            total_generations=10,
            initial_batch_size=4,
        )
        # History is populated (so histogram renders) but _decisions is empty.
        controller.record_yield(GenerationYield(generation=1, batch_size=4, successes=2))
        controller.record_yield(GenerationYield(generation=2, batch_size=3, successes=1))
        assert controller._decisions == []

        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=False,
        )
        output = "\n".join(lines)
        assert "Shrink events: **0**" in output
        assert "Expand events: **0**" in output
        # Histogram still renders from _history.
        assert "### Effective-batch histogram" in output
        assert "| 3 | 1 |" in output
        assert "| 4 | 1 |" in output

    def test_histogram_ascending_order(self):
        import re

        from esn.engine.batch_budget import (
            BatchBudgetController,
            GenerationYield,
        )
        from esn.engine.batch_budget_report import render_adaptive_batching_section

        controller = BatchBudgetController(
            total_generations=10,
            initial_batch_size=4,
        )
        # Deliberately out-of-order batch sizes with duplicates.
        for gen, bsz in enumerate([5, 3, 4, 3, 5, 5], start=1):
            controller.record_yield(GenerationYield(generation=gen, batch_size=bsz, successes=0))

        lines = render_adaptive_batching_section(
            controller,
            nominal_batch_size=4,
            resume_restored_batch_size=False,
        )

        # Locate the histogram section and collect size-column values
        # from each data row.
        header_idx = lines.index("### Effective-batch histogram")
        row_re = re.compile(r"^\| (\d+) \| \d+ \|$")
        sizes: list[int] = []
        for line in lines[header_idx + 1 :]:
            m = row_re.match(line)
            if m:
                sizes.append(int(m.group(1)))

        # Unique sizes from input: {3, 4, 5}; expect ascending, no dups.
        assert sizes == sorted(sizes)
        assert len(sizes) == len(set(sizes))
        assert sizes == [3, 4, 5]
