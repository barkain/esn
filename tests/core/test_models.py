"""Tests for ESN core data models."""

from __future__ import annotations

import time

from esn.core.enums import SearchMode
from esn.core.models import (
    CandidateRecord,
    CompilerResult,
    EvaluationDiagnostics,
    EvaluationResult,
    ImprovementContext,
    ImprovementResult,
    MutationContext,
    MutationPlan,
    MutationResult,
    OperatorStats,
    SearchState,
)


class TestCompilerResult:
    def test_defaults(self):
        r = CompilerResult(artifact="out.bin", success=True)
        assert r.errors == []
        assert r.warnings == []
        assert r.metadata == {}

    def test_all_fields(self):
        r = CompilerResult(artifact="out.bin", success=False, errors=["e1"], warnings=["w1"])
        assert r.errors == ["e1"]
        assert not r.success


class TestEvaluationDiagnostics:
    def test_defaults(self):
        d = EvaluationDiagnostics()
        assert d.constraints == {}
        assert d.violations == []
        assert d.residuals == {}
        assert d.complexity == {}
        assert d.robustness == {}
        assert d.resources == {}
        assert d.notes == []

    def test_independent_defaults(self):
        d1 = EvaluationDiagnostics()
        d2 = EvaluationDiagnostics()
        d1.violations.append("v")
        assert d2.violations == []

    def test_all_fields(self):
        d = EvaluationDiagnostics(
            constraints={"c": 1},
            violations=["v"],
            residuals={"r": 0.1},
            complexity={"cx": 2.0},
            robustness={"rb": 0.9},
            resources={"mem": 100.0},
            notes=["note"],
        )
        assert d.constraints == {"c": 1}


class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult(score=0.5, success=True)
        assert r.diagnostics is None
        assert r.raw_outputs == {}

    def test_with_diagnostics(self):
        diag = EvaluationDiagnostics(violations=["v1"])
        r = EvaluationResult(score=1.0, success=True, diagnostics=diag)
        assert r.diagnostics.violations == ["v1"]


class TestMutationPlan:
    def test_defaults(self):
        p = MutationPlan(search_mode=SearchMode.EXPLOIT, operator_name="swap")
        assert p.target == ""
        assert p.risk == "low"
        assert p.parameters == {}

    def test_all_fields(self):
        p = MutationPlan(
            search_mode=SearchMode.EXPLORE,
            operator_name="insert",
            target="fn_a",
            parameters={"k": 1},
            rationale="test",
            expected_effect="improve",
            risk="high",
        )
        assert p.risk == "high"


class TestMutationContext:
    def test_defaults(self):
        c = MutationContext(search_mode=SearchMode.REPAIR)
        assert c.parent_summary == ""
        assert c.top_hypotheses == []


class TestMutationResult:
    def test_defaults(self):
        r = MutationResult(success=True)
        assert r.mutated_object is None
        assert r.errors == []


class TestImprovementModels:
    def test_improvement_context_defaults(self):
        c = ImprovementContext(search_mode=SearchMode.EXPLOIT)
        assert c.budget == 1
        assert c.metadata == {}

    def test_improvement_result_defaults(self):
        r = ImprovementResult(success=True)
        assert r.changed is False
        assert r.errors == []


class TestCandidateRecord:
    def test_created_at_auto(self):
        before = time.time()
        rec = CandidateRecord(
            id="c1",
            generation=0,
            search_mode=SearchMode.EXPLOIT,
            operator_name="noop",
            object_hash="abc",
        )
        after = time.time()
        assert before <= rec.created_at <= after

    def test_optional_fields(self):
        rec = CandidateRecord(
            id="c2",
            generation=1,
            search_mode=SearchMode.BRIDGE,
            operator_name="op",
            object_hash="def",
            score=0.9,
            epistemic_novelty=0.5,
        )
        assert rec.score == 0.9
        assert rec.parent_id is None
        assert rec.object_summary == ""
        assert rec.plan_rationale == ""
        assert rec.plan_expected_effect == ""
        assert rec.compiled_artifact == ""
        assert rec.realized_artifact_summary == ""
        assert rec.compile_metadata == {}


class TestOperatorStats:
    def test_defaults(self):
        s = OperatorStats()
        assert s.attempts == 0
        assert s.mean_score_delta == 0.0
        assert s.recent_score_delta == 0.0
        assert s.non_improving_streak == 0


class TestSearchState:
    def test_defaults(self):
        s = SearchState()
        assert s.generation == 0
        assert s.current_mode == SearchMode.EXPLOIT
        assert s.recent_scores == []
        assert s.recent_operators == []
        assert s.frontier_distinct_count == 0

    def test_serialization_roundtrip(self):
        s = SearchState(generation=5, best_score=1.5, current_mode=SearchMode.EXPLORE)
        data = s.model_dump()
        s2 = SearchState.model_validate(data)
        assert s2.generation == 5
        assert s2.current_mode == SearchMode.EXPLORE


class TestSerializationRoundtrip:
    def test_candidate_record(self):
        rec = CandidateRecord(
            id="c1",
            generation=3,
            search_mode=SearchMode.COMPRESS,
            operator_name="prune",
            object_hash="xyz",
            score=0.8,
        )
        data = rec.model_dump()
        rec2 = CandidateRecord.model_validate(data)
        assert rec2.id == "c1"
        assert rec2.search_mode == SearchMode.COMPRESS

    def test_evaluation_result(self):
        r = EvaluationResult(
            score=0.7,
            success=True,
            diagnostics=EvaluationDiagnostics(violations=["v"]),
        )
        data = r.model_dump()
        r2 = EvaluationResult.model_validate(data)
        assert r2.diagnostics.violations == ["v"]
