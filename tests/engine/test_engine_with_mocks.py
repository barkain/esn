"""Tests for ESNEngine with mock mutator/predictor/analyzer."""

from __future__ import annotations

from esn.core.models import EvaluationResult
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine
from esn.engine.models import AnalysisResult, MutationResult, PredictionResult


INITIAL_CODE = "def solve():\n    return [1, 2, 3]\n"


def _sum_evaluator(artifact):
    if artifact is None:
        return EvaluationResult(score=0.0, success=False)
    try:
        return EvaluationResult(score=float(sum(artifact)), success=True)
    except Exception:
        return EvaluationResult(score=0.0, success=False)


def _make_domain():
    return DomainSpec(
        name="mock-test",
        description="test domain for mock tests",
        initial_code=INITIAL_CODE,
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=_sum_evaluator,
    )


class MockMutator:
    def __init__(self, new_code: str):
        self._code = new_code
        self.call_count = 0
        self.last_parents = None
        self.last_style = None
        self.last_context = None

    def mutate(self, parents, style, context):
        self.call_count += 1
        self.last_parents = parents
        self.last_style = style
        self.last_context = context
        return MutationResult(code=self._code, success=True)


class MockFailingMutator:
    def __init__(self):
        self.call_count = 0

    def mutate(self, parents, style, context):
        self.call_count += 1
        return MutationResult(code="", success=False, errors=["mutation failed"])


class MockPredictor:
    def __init__(self, score_range=(0.0, 10.0)):
        self._range = score_range
        self.call_count = 0

    def predict(self, program, mutation_style, hypotheses, score_history):
        self.call_count += 1
        return PredictionResult(score_range=self._range)


class MockAnalyzer:
    def __init__(self):
        self.call_count = 0

    def analyze(self, solution_summary, score, diagnostics, active_hypotheses, strategy):
        self.call_count += 1
        return AnalysisResult()


class CountingEmbedder:
    """Stand-in embedder that counts how many times it was invoked."""

    def __init__(self):
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return [0.1, 0.2, 0.3]


class MockKnowledge:
    """Minimal stand-in for KnowledgeIntegration."""

    def __init__(self, embedder=None):
        self.preview_call_count = 0
        self.apply_call_count = 0
        self.maintenance_call_count = 0
        self._embedder = embedder

    def get_active_hypotheses_for_prompt(self, limit=10, novelty_computer=None):
        return [{"id": "h1", "text": "test hypothesis", "confidence": 0.5}]

    def preview_analysis(self, analysis, generation, enrichment=None):
        self.preview_call_count += 1
        self.last_enrichment = enrichment
        return {"relevant_data": [], "new_count": 0, "engaged": []}

    def apply_prepared_analysis(self, analysis_data, generation):
        self.apply_call_count += 1

    def run_maintenance(self, generation):
        self.maintenance_call_count += 1


class TestMutatorIntegration:
    def test_mutator_called_with_parents(self):
        better_code = "def solve():\n    return [10, 20, 30]\n"
        mutator = MockMutator(better_code)
        engine = ESNEngine(domain=_make_domain(), mutator=mutator)
        engine.run_generation()
        assert mutator.call_count == 1

    def test_mutator_receives_style(self):
        mutator = MockMutator(INITIAL_CODE)
        engine = ESNEngine(domain=_make_domain(), mutator=mutator)
        engine.run_generation()
        assert mutator.last_style is not None
        assert isinstance(mutator.last_style, str)

    def test_mutation_failure_handled(self):
        mutator = MockFailingMutator()
        engine = ESNEngine(domain=_make_domain(), mutator=mutator)
        record = engine.run_generation()
        assert record.success is False


class TestPredictorIntegration:
    def test_predictor_called_when_knowledge_present(self):
        predictor = MockPredictor()
        knowledge = MockKnowledge()
        engine = ESNEngine(
            domain=_make_domain(),
            predictor=predictor,
            knowledge=knowledge,
        )
        engine.run_generation()
        assert predictor.call_count == 1

    def test_predictor_not_called_without_knowledge(self):
        predictor = MockPredictor()
        engine = ESNEngine(
            domain=_make_domain(),
            predictor=predictor,
            knowledge=None,
        )
        engine.run_generation()
        assert predictor.call_count == 0


class TestAnalyzerIntegration:
    def test_analyzer_called_when_knowledge_present(self):
        analyzer = MockAnalyzer()
        knowledge = MockKnowledge()
        engine = ESNEngine(
            domain=_make_domain(),
            analyzer=analyzer,
            knowledge=knowledge,
        )
        engine.run_generation()
        assert analyzer.call_count == 1

    def test_knowledge_preview_before_apply(self):
        analyzer = MockAnalyzer()
        knowledge = MockKnowledge()
        engine = ESNEngine(
            domain=_make_domain(),
            analyzer=analyzer,
            knowledge=knowledge,
        )
        engine.run_generation()
        # Both preview and apply should have been called
        assert knowledge.preview_call_count == 1
        assert knowledge.apply_call_count == 1


class MockLocalImprover:
    """Local improver that returns a fixed code with a higher score."""

    def __init__(self, improved_code: str, improved_score: float):
        self._code = improved_code
        self._score = improved_score
        self.call_count = 0

    def improve(self, code, artifact, score, evaluator):
        from esn.engine.local_improver import LocalImprovementResult

        self.call_count += 1
        return LocalImprovementResult(
            improved=True,
            code=self._code,
            artifact=None,
            score=self._score,
        )


class TestScoreImprovement:
    def test_score_improvement_updates_best(self):
        # Gen 1: identity code scores 6.0
        # Gen 2: mutator returns code scoring 60.0
        better_code = "def solve():\n    return [10, 20, 30]\n"
        mutator = MockMutator(better_code)
        engine = ESNEngine(domain=_make_domain(), mutator=mutator)

        # Gen 1: mutator produces better_code (scores 60)
        record = engine.run_generation()
        assert record.score == 60.0
        assert engine._best_score == 60.0
        assert engine._best_code == better_code


class TestBranchSignalsSkipEmbedding:
    """Issue #9: failure-path ``_branch_signals`` must skip embedding
    inference while still returning a valid aspect_signature so
    BranchManager can account for the failed attempt."""

    def test_skip_flag_returns_signature_without_calling_embedder(self):
        embedder = CountingEmbedder()
        knowledge = MockKnowledge(embedder=embedder)
        engine = ESNEngine(
            domain=_make_domain(),
            mutator=MockMutator(INITIAL_CODE),
            knowledge=knowledge,
        )
        sig, emb = engine._branch_signals(
            "def f():\n    return 1\n", "grid", compute_embedding=False
        )
        assert isinstance(sig, str) and sig
        assert emb is None
        assert embedder.call_count == 0

    def test_default_still_computes_embedding(self):
        engine = ESNEngine(
            domain=_make_domain(),
            mutator=MockMutator(INITIAL_CODE),
        )
        sig, emb = engine._branch_signals("def f():\n    return 1\n", "grid")
        assert isinstance(sig, str) and sig
        assert isinstance(emb, list) and len(emb) > 0
        # Feature vector is L2-normalised
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_failure_path_does_not_compute_embedding(self):
        """End-to-end: after the seed success call, generations that fail
        every candidate must not compute embeddings for failed attempts."""
        engine = ESNEngine(
            domain=_make_domain(),
            mutator=MockFailingMutator(),
        )
        # First generation seeds the founder and runs a failing mutation.
        engine.run_generation()
        baseline_cache_size = len(engine._aspect_embedding_cache)
        # Subsequent generations consist entirely of failing mutations —
        # these must NOT compute embeddings (issue #9: compute_embedding=False).
        for _ in range(3):
            engine.run_generation()
        assert len(engine._aspect_embedding_cache) == baseline_cache_size, (
            f"failure path should not compute embeddings; "
            f"cache grew from {baseline_cache_size} to {len(engine._aspect_embedding_cache)}"
        )


class TestPolishShearFamilyReclassification:
    """Polish-shear fix: when the local improver rewrites outcome.new_code,
    the family classification must reflect the post-polish code, not the
    pre-polish LLM output."""

    def test_family_reflects_post_polish_code(self):
        from esn.engine.ast_features import extract_ast_features

        # Pre-polish code is straight-line (no loops, no recursion).
        pre_polish = INITIAL_CODE  # "def solve(): return [1, 2, 3]"
        assert extract_ast_features(pre_polish)["family"] == "straight-line"

        # Post-polish code has a for loop → iterative-flat.
        post_polish = (
            "def solve():\n"
            "    r = []\n"
            "    for i in range(3):\n"
            "        r.append(i * 10)\n"
            "    return r\n"
        )
        assert extract_ast_features(post_polish)["family"] == "iterative-flat"

        # MockMutator returns the same straight-line code as INITIAL_CODE.
        # MockLocalImprover rewrites it to the iterative-flat code with
        # a higher score. If the fix works, the candidate's family should
        # be "iterative-flat" (post-polish), not "straight-line" (pre-polish).
        mutator = MockMutator(pre_polish)
        improver = MockLocalImprover(improved_code=post_polish, improved_score=99.0)
        engine = ESNEngine(
            domain=_make_domain(),
            mutator=mutator,
            local_improver=improver,
        )
        record = engine.run_generation()
        assert record.success is True
        assert record.family == "iterative-flat", (
            f"expected post-polish family 'iterative-flat', got '{record.family}'"
        )
