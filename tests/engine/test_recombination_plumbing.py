"""Engine-level plumbing tests for recombination.

These tests bypass the LLM-serendipity problem (naturally producing >= 3
live branches in a short run is hard) by handcrafting a branch manager
state that passes every gate, then exercising `_plan_batch_legacy` and
`_process_outcome` directly.
"""

from __future__ import annotations

from esn.core.enums import SearchMode
from esn.core.models import EvaluationResult
from esn.engine.branch_manager import MIN_PLATEAU_FOR_RECOMBINE
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine, _CandidateOutcome


def _evaluator(artifact):
    if artifact is None:
        return EvaluationResult(score=0.0, success=False)
    try:
        return EvaluationResult(score=float(sum(artifact)), success=True)
    except Exception:
        return EvaluationResult(score=0.0, success=False)


def _make_domain() -> DomainSpec:
    return DomainSpec(
        name="test",
        description="recombination plumbing",
        initial_code="def solve():\n    return [1, 2, 3]\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=_evaluator,
        allowed_imports=frozenset({"math"}),
    )


def _seed_three_branches(engine: ESNEngine) -> tuple[str, str, str]:
    """Populate branch manager with 3 diverse high-score branches.

    Returns (anchor_branch_id, donor_branch_id, other_id).
    """
    codes = {
        "anchor_code": "def solve():\n    return [100, 100, 100]\n",
        "donor_code": "def solve():\n    return [90, 90, 90]\n",
        "other_code": "def solve():\n    return [85, 85, 85]\n",
    }
    program_ids = {}
    for label, code in codes.items():
        pid = f"pid_{label}"
        engine._program_store[pid] = code
        program_ids[label] = pid

    embeddings = {
        "anchor_code": [1.0, 0.0, 0.0],
        "donor_code": [0.0, 1.0, 0.0],
        "other_code": [0.0, 0.0, 1.0],
    }
    families = {
        "anchor_code": "grid",
        "donor_code": "hex",
        "other_code": "spiral",
    }
    scores = {
        "anchor_code": 300.0,
        "donor_code": 270.0,
        "other_code": 255.0,
    }

    branch_ids: dict[str, str] = {}
    for label in ("anchor_code", "donor_code", "other_code"):
        a = engine._branch_manager.register_attempt(
            parent_id=None,
            child_id=program_ids[label],
            success=True,
            score=scores[label],
            family=families[label],
            aspect_signature=f"family={families[label]}",
            generation=0,
            embedding=embeddings[label],
        )
        branch_ids[label] = a.branch_id

    return (
        branch_ids["anchor_code"],
        branch_ids["donor_code"],
        branch_ids["other_code"],
    )


class TestPlanBatchRecombineAllocation:
    def test_recombine_slot_replaces_one_explore_when_gates_pass(self):
        engine = ESNEngine(
            domain=_make_domain(),
            batch_size=3,
            enable_recombination=True,
        )
        engine.generation = 20
        engine._best_score = 300.0
        engine.state.stagnation_counter = MIN_PLATEAU_FOR_RECOMBINE + 1
        anchor_id, donor_id, _ = _seed_three_branches(engine)
        engine._best_code = engine._program_store[
            engine._branch_manager.branches[anchor_id].latest_parent_id
        ]

        plan = engine._plan_batch(SearchMode.EXPLORE)

        # At most one recombine, and when present it must carry two parents
        recomb_slots = [i for i, (ps, s) in enumerate(plan) if s == "recombine"]
        assert len(recomb_slots) == 1, (
            f"expected exactly 1 recombine slot, got plan={[(len(p), s) for p, s in plan]}"
        )
        slot_idx = recomb_slots[0]
        parents, style = plan[slot_idx]
        assert style == "recombine"
        assert len(parents) == 2
        # Slot 0 is never replaced
        assert slot_idx != 0
        # Donor metadata cached for provenance
        assert slot_idx in engine._recomb_slot_meta
        meta = engine._recomb_slot_meta[slot_idx]
        assert meta["anchor_branch_id"] == anchor_id
        assert meta["donor_branch_id"] == donor_id

    def test_recombine_disabled_flag_skips_allocation(self):
        engine = ESNEngine(
            domain=_make_domain(),
            batch_size=3,
            enable_recombination=False,
        )
        engine.generation = 20
        engine._best_score = 300.0
        engine.state.stagnation_counter = MIN_PLATEAU_FOR_RECOMBINE + 1
        _seed_three_branches(engine)

        plan = engine._plan_batch(SearchMode.EXPLORE)
        assert all(s != "recombine" for _p, s in plan)
        assert engine._recomb_slot_meta == {}

    def test_recombine_skipped_when_gates_fail(self):
        engine = ESNEngine(
            domain=_make_domain(),
            batch_size=3,
            enable_recombination=True,
        )
        engine.generation = 20
        engine._best_score = 300.0
        # Stagnation too low → plateau gate fails
        engine.state.stagnation_counter = 0
        _seed_three_branches(engine)

        plan = engine._plan_batch(SearchMode.EXPLORE)
        assert all(s != "recombine" for _p, s in plan)


class TestProcessOutcomeRecombinationProvenance:
    def _make_outcome(
        self,
        engine: ESNEngine,
        slot: int,
        parent_code: str,
        new_code: str,
        score: float,
        success: bool,
    ) -> _CandidateOutcome:
        from esn.core.models import EvaluationDiagnostics
        from esn.engine.models import MutationContext

        oc = _CandidateOutcome(
            slot=slot,
            style="recombine",
            mode=SearchMode.EXPLORE,
            parent_code=parent_code,
            context=MutationContext(search_mode="explore", mutation_style="recombine"),
        )
        oc.new_code = new_code
        oc.score = score
        oc.raw_score = score
        oc.success = success
        oc.family = "grid"
        oc.family_confidence = "high"
        oc.code_hash = "hash_" + str(slot)
        oc.eval_result = EvaluationResult(
            score=score,
            success=success,
            diagnostics=EvaluationDiagnostics(),
        )
        return oc

    def test_successful_recombine_credits_donor_and_stamps_provenance(self):
        engine = ESNEngine(
            domain=_make_domain(),
            batch_size=3,
            enable_recombination=True,
        )
        engine.generation = 20
        engine._best_score = 300.0
        anchor_id, donor_id, _ = _seed_three_branches(engine)

        # Manually stamp slot meta as _plan_batch would
        engine._recomb_slot_meta[1] = {
            "anchor_branch_id": anchor_id,
            "donor_branch_id": donor_id,
        }

        anchor_code = engine._program_store[
            engine._branch_manager.branches[anchor_id].latest_parent_id
        ]
        outcome = self._make_outcome(
            engine,
            slot=1,
            parent_code=anchor_code,
            new_code="def solve():\n    return [1,2,3]\n",
            score=310.0,
            success=True,
        )
        record = engine._process_outcome(outcome)

        # Provenance stamped
        assert "recombined_from" in record.compile_metadata
        assert record.compile_metadata["recombined_from"] == [anchor_id, donor_id]

        # Donor counters incremented on anchor-parent branch for recomb_donor
        donor = engine._branch_manager.branches[donor_id]
        assert donor.num_donor_attempts == 1
        assert donor.num_donor_successes == 1
        # Donor primary counters untouched
        assert donor.num_successes == 1  # only the seed
        assert donor.num_improvements == 1

        # Event logged
        events = engine._branch_manager.recombination_events
        assert len(events) == 1
        assert events[0]["success"] is True
        assert events[0]["anchor_branch_id"] == anchor_id
        assert events[0]["donor_branch_id"] == donor_id

        # Cooldown updated
        assert engine._branch_manager.last_recombination_gen == 20

    def test_failed_recombine_still_credits_donor_attempt(self):
        engine = ESNEngine(
            domain=_make_domain(),
            batch_size=3,
            enable_recombination=True,
        )
        engine.generation = 20
        engine._best_score = 300.0
        anchor_id, donor_id, _ = _seed_three_branches(engine)
        engine._recomb_slot_meta[1] = {
            "anchor_branch_id": anchor_id,
            "donor_branch_id": donor_id,
        }

        anchor_code = engine._program_store[
            engine._branch_manager.branches[anchor_id].latest_parent_id
        ]
        outcome = self._make_outcome(
            engine,
            slot=1,
            parent_code=anchor_code,
            new_code="def solve():\n    return [1,2,3]\n",
            score=0.0,
            success=False,
        )
        # Force an eval failure via the eval-fail success path
        record = engine._process_outcome(outcome)

        donor = engine._branch_manager.branches[donor_id]
        assert donor.num_donor_attempts == 1
        assert donor.num_donor_successes == 0

        events = engine._branch_manager.recombination_events
        assert len(events) == 1
        assert events[0]["success"] is False
        # Provenance still stamped on failed record
        assert "recombined_from" in record.compile_metadata
