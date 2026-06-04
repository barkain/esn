"""Tests for engine LLM mutator / predictor / analyzer components."""

from __future__ import annotations

import json

import pytest

from esn.core.llm_adapters import LLMAPIError, MockLLMClient
from esn.engine import LLMAnalyzer, LLMMutator, LLMPredictor
from esn.engine.domain import DomainSpec
from esn.engine.engine import _CodeWrapper
from esn.engine.models import MutationContext


def _domain() -> DomainSpec:
    from esn.engine.compiler import PythonSandboxCompiler
    from esn.core.models import EvaluationResult

    return DomainSpec(
        name="test",
        description="simple test domain",
        initial_code="def solve():\n    return [1, 2, 3]\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=lambda artifact: EvaluationResult(score=1.0, success=True),
        allowed_imports=frozenset({"math"}),
        max_code_lines=40,
        hard_constraints=["must define solve()"],
        hints=["keep changes small"],
    )


def _domain_with_time() -> DomainSpec:
    """Variant of ``_domain()`` whose allowlist includes ``time``.

    Used by tests that exercise the concrete ``time.time()`` branch of the
    domain-aware runtime-budget hint.
    """
    from esn.engine.compiler import PythonSandboxCompiler
    from esn.core.models import EvaluationResult

    return DomainSpec(
        name="test",
        description="simple test domain",
        initial_code="def solve():\n    return [1, 2, 3]\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math", "time"})),
        evaluator=lambda artifact: EvaluationResult(score=1.0, success=True),
        allowed_imports=frozenset({"math", "time"}),
        max_code_lines=40,
        hard_constraints=["must define solve()"],
        hints=["keep changes small"],
    )


class TestLLMMutator:
    def test_mutator_accepts_raw_code(self):
        client = MockLLMClient("def solve():\n    return [2, 3, 4]\n")
        mutator = LLMMutator(client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is True
        assert "def solve" in result.code

    def test_mutator_parses_json_metadata(self):
        client = MockLLMClient(
            json.dumps(
                {
                    "code": "def solve():\n    return [3, 4, 5]\n",
                    "diff_summary": "changed constants",
                    "intended_effect": "increase score",
                }
            )
        )
        mutator = LLMMutator(client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is True
        assert result.metadata["diff_summary"] == "changed constants"

    def test_mutator_rejects_invalid_code(self):
        client = MockLLMClient("import os\ndef solve():\n    return 1\n")
        mutator = LLMMutator(client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is False
        assert any("Disallowed import" in error for error in result.errors)

    def test_mutator_retries_on_empty_response(self):
        """Phase 0.2: empty LLM responses must trigger a retry with feedback."""

        responses = [
            "",  # first attempt: empty
            "def solve():\n    return [9, 9, 9]\n",  # second attempt: valid
        ]
        calls: list[tuple[str, str]] = []

        def _client(system_prompt: str, user_prompt: str) -> str:
            calls.append((system_prompt, user_prompt))
            return responses[len(calls) - 1]

        mutator = LLMMutator(_client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is True
        assert "[9, 9, 9]" in result.code
        assert len(calls) == 2
        # Second call must include the explicit failure feedback
        assert "PREVIOUS ATTEMPT FAILED" in calls[1][1]
        assert result.metadata["mutator_attempts"] == 2

    def test_mutator_retries_on_validation_failure(self):
        """AST validation failures should also trigger retry with feedback."""

        responses = [
            "import os\ndef solve():\n    return 1\n",  # disallowed import
            "import os\ndef solve():\n    return 1\n",  # still bad
            "def solve():\n    return [7, 7, 7]\n",  # finally valid
        ]
        calls: list[tuple[str, str]] = []

        def _client(system_prompt: str, user_prompt: str) -> str:
            calls.append((system_prompt, user_prompt))
            return responses[len(calls) - 1]

        mutator = LLMMutator(_client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is True
        assert "[7, 7, 7]" in result.code
        assert len(calls) == 3
        assert result.metadata["mutator_attempts"] == 3
        assert "PREVIOUS ATTEMPT FAILED" in calls[1][1]
        assert "Disallowed import" in calls[1][1] or "os" in calls[1][1]

    def test_mutator_strips_trailing_json_metadata_suffix(self):
        """Raw code followed by JSON metadata should not be stored as code."""

        response = """def solve():
    return [4, 5, 6]

{"code": "...", "diff_summary": "changed constants", "intended_effect": "improve score"}
"""
        mutator = LLMMutator(MockLLMClient(response), _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is True
        assert result.code.strip().endswith("return [4, 5, 6]")
        assert '{"code":' not in result.code
        assert result.metadata["diff_summary"] == "changed constants"
        assert result.metadata["intended_effect"] == "improve score"

    def test_mutator_gives_up_after_max_attempts(self):
        """After 3 failed attempts, mutator returns failure with last error."""

        calls: list[int] = []

        def _client(system_prompt: str, user_prompt: str) -> str:
            calls.append(1)
            return ""  # always empty

        mutator = LLMMutator(_client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is False
        assert len(calls) == 3
        assert "empty" in result.errors[0].lower() or "failed" in result.errors[0].lower()
        assert result.metadata.get("mutator_attempts") == 3

    def test_mutator_prompt_includes_multiple_parents_for_synthesize(self):
        client = MockLLMClient("def solve():\n    return [1, 2, 3]\n")
        mutator = LLMMutator(client, _domain())
        mutator.mutate(
            [
                _CodeWrapper("def solve():\n    return [1]\n"),
                _CodeWrapper("def solve():\n    return [2]\n"),
            ],
            "synthesize",
            MutationContext(search_mode="bridge", mutation_style="synthesize"),
        )
        assert "Parent 1:" in client.last_user_prompt
        assert "Parent 2:" in client.last_user_prompt

    def test_mutator_system_prompt_includes_wallclock_runtime_hint(self):
        """The mutator's system prompt must tell candidates to install an
        explicit elapsed-time guard on long-running loops — but only when
        the domain allows importing ``time``.

        Regression test: weak models (e.g. Haiku) produce solutions whose
        outer search / anneal loops have no wall-clock guard, so they get
        killed by the harness-enforced cap with nothing to score. The hint
        is a short, benchmark-agnostic paragraph. For domains whose
        ``allowed_imports`` excludes ``time``, the concrete ``time.time()``
        recommendation would induce AST-validator rejection, so a generic
        iteration-cap variant is rendered instead.
        """
        # Branch 1: domain allows `time` -> concrete time.time() template.
        client = MockLLMClient("def solve():\n    return [1, 2, 3]\n")
        mutator = LLMMutator(client, _domain_with_time())
        mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        sys_prompt = client.last_system_prompt
        assert "time.time()" in sys_prompt, (
            "mutator system prompt (time-allowed domain) is missing the "
            f"'time.time()' concrete hint. Got:\n{sys_prompt}"
        )
        assert "wall-clock" in sys_prompt.lower()

    def test_mutator_system_prompt_uses_generic_runtime_hint_when_time_disallowed(self):
        """When ``allowed_imports`` excludes ``time``, the hint must NOT
        name ``time.time()`` or any timing API — otherwise the post-mutation
        AST validator would reject exactly what the hint induces.
        """
        client = MockLLMClient("def solve():\n    return [1, 2, 3]\n")
        mutator = LLMMutator(client, _domain())  # allowed_imports = {"math"}
        mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        sys_prompt = client.last_system_prompt
        assert "time.time" not in sys_prompt, (
            "mutator system prompt (time-disallowed domain) must NOT name "
            f"time.time(). Got:\n{sys_prompt}"
        )
        assert "iteration cap" in sys_prompt.lower(), (
            "mutator system prompt (time-disallowed domain) must carry the "
            f"generic iteration-cap variant. Got:\n{sys_prompt}"
        )


class TestLLMPredictor:
    def test_predictor_parses_score_range_and_ids(self):
        client = MockLLMClient(
            json.dumps(
                {
                    "score_range": [1.0, 2.5],
                    "relevant_hypothesis_ids": ["h1", "missing"],
                    "reasoning": "small improvement likely",
                }
            )
        )
        predictor = LLMPredictor(client)
        result = predictor.predict(
            _CodeWrapper("def solve():\n    return 1\n"),
            "refine",
            [{"id": "h1", "text": "test"}],
            {"best": 1.0},
        )
        assert result.score_range == (1.0, 2.5)
        assert result.relevant_hypothesis_ids == ["h1"]

    def test_predictor_falls_back_on_invalid_json(self):
        predictor = LLMPredictor(MockLLMClient("not json"))
        result = predictor.predict(
            _CodeWrapper("def solve():\n    return 1\n"),
            "refine",
            [],
            {},
        )
        assert result.score_range == (0.0, 1.0)


class TestLLMAnalyzer:
    def test_analyzer_parses_evidence_and_new_hypotheses(self):
        client = MockLLMClient(
            json.dumps(
                {
                    "evidence": [
                        {"hypothesis_id": "h1", "evidence": 1, "explanation": "supported"},
                        {"hypothesis_id": "missing", "evidence": 0, "explanation": "ignored"},
                    ],
                    "new_hypotheses": [
                        {"text": "try larger moves", "concepts": ["moves", "exploration"]}
                    ],
                }
            )
        )
        analyzer = LLMAnalyzer(client)
        result = analyzer.analyze(
            solution_summary="def solve(): return 1",
            score=1.0,
            diagnostics={},
            active_hypotheses=[{"id": "h1", "text": "current idea", "confidence": 0.5}],
            strategy="refine",
        )
        assert len(result.evidence) == 1
        assert result.evidence[0].hypothesis_id == "h1"
        assert len(result.new_hypotheses) == 1

    def test_analyzer_falls_back_on_invalid_json(self):
        analyzer = LLMAnalyzer(MockLLMClient("bad json"))
        result = analyzer.analyze(
            solution_summary="def solve(): return 1",
            score=1.0,
            diagnostics={},
            active_hypotheses=[],
            strategy="refine",
        )
        assert result.evidence == []
        assert result.new_hypotheses == []


class TestLLMAPIErrorPropagation:
    """LLMAPIError must propagate through the mutator catch-all."""

    def test_mutator_propagates_llm_api_error(self):
        """LLMAPIError from the LLM client must NOT be caught by the mutator."""

        def _failing_client(_sys: str, _usr: str) -> str:
            raise LLMAPIError("OpenAI API fatal error: quota exceeded")

        mutator = LLMMutator(_failing_client, _domain())
        with pytest.raises(LLMAPIError, match="quota exceeded"):
            mutator.mutate(
                [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
                "refine",
                MutationContext(search_mode="exploit", mutation_style="refine"),
            )

    def test_mutator_still_catches_normal_exceptions(self):
        """Non-API exceptions should still be caught and returned as failures."""

        def _failing_client(_sys: str, _usr: str) -> str:
            raise ValueError("some transient parse error")

        mutator = LLMMutator(_failing_client, _domain())
        result = mutator.mutate(
            [_CodeWrapper("def solve():\n    return [1, 2, 3]\n")],
            "refine",
            MutationContext(search_mode="exploit", mutation_style="refine"),
        )
        assert result.success is False
        assert "transient parse error" in result.errors[0]
