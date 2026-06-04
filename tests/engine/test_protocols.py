"""Protocol conformance tests for ESN engine."""

from __future__ import annotations

from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import _CodeWrapper
from esn.engine.models import AnalysisResult, MutationResult, PredictionResult
from esn.engine.protocols import Analyzer, Mutator, Predictor, ProgramCompiler, ProgramObject


class TestCodeWrapperProtocol:
    def test_code_wrapper_satisfies_program_object(self):
        wrapper = _CodeWrapper("def solve():\n    return 1\n")
        assert isinstance(wrapper, ProgramObject)

    def test_code_wrapper_code_property(self):
        wrapper = _CodeWrapper("def solve():\n    return 1\n")
        assert wrapper.code == "def solve():\n    return 1\n"

    def test_code_wrapper_summary(self):
        wrapper = _CodeWrapper("def solve():\n    return 1\n")
        assert "def solve" in wrapper.summary()

    def test_code_wrapper_structural_hash(self):
        w1 = _CodeWrapper("code_a")
        w2 = _CodeWrapper("code_a")
        w3 = _CodeWrapper("code_b")
        assert w1.structural_hash() == w2.structural_hash()
        assert w1.structural_hash() != w3.structural_hash()

    def test_code_wrapper_serialize_deserialize(self):
        original = _CodeWrapper("def solve():\n    return 42\n")
        data = original.serialize()
        restored = _CodeWrapper.deserialize(data)
        assert restored.code == original.code


class TestCompilerProtocol:
    def test_sandbox_compiler_satisfies_protocol(self):
        compiler = PythonSandboxCompiler()
        assert isinstance(compiler, ProgramCompiler)


class TestMockProtocolConformance:
    def test_mock_mutator_satisfies_protocol(self):
        class _Mut:
            def mutate(self, parents, style, context):
                return MutationResult(code="", success=True)

        assert isinstance(_Mut(), Mutator)

    def test_mock_predictor_satisfies_protocol(self):
        class _Pred:
            def predict(self, program, mutation_style, hypotheses, score_history):
                return PredictionResult()

        assert isinstance(_Pred(), Predictor)

    def test_mock_analyzer_satisfies_protocol(self):
        class _Ana:
            def analyze(self, solution_summary, score, diagnostics, active_hypotheses, strategy):
                return AnalysisResult()

        assert isinstance(_Ana(), Analyzer)


class TestDomainSpec:
    def test_domain_spec_construction(self):
        from esn.core.models import EvaluationResult

        domain = DomainSpec(
            name="test",
            description="a test domain",
            initial_code="def solve(): return 1",
            compiler=PythonSandboxCompiler(),
            evaluator=lambda x: EvaluationResult(score=1.0, success=True),
            allowed_imports=frozenset({"math"}),
            max_code_lines=300,
            hard_constraints=["no overlap"],
            examples=["example 1"],
            hints=["try greedy"],
        )
        assert domain.name == "test"
        assert domain.max_code_lines == 300
        assert "math" in domain.allowed_imports

    def test_domain_spec_defaults(self):
        from esn.core.models import EvaluationResult

        domain = DomainSpec(
            name="minimal",
            description="minimal",
            initial_code="def solve(): return 1",
            compiler=PythonSandboxCompiler(),
            evaluator=lambda x: EvaluationResult(score=0.0, success=True),
        )
        assert domain.allowed_imports == frozenset()
        assert domain.max_code_lines is None
        assert domain.hard_constraints == []
        assert domain.examples == []
        assert domain.hints == []
