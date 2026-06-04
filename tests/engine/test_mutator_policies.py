"""Tests for engine mutator policy dispatch (single_shot vs agentic_v1)."""

from __future__ import annotations

import pytest

from esn.core.llm_adapters import LLMAPIError
from esn.engine import LLMMutator
from esn.engine.domain import DomainSpec
from esn.engine.engine import _CodeWrapper
from esn.engine.models import MutationContext, MutationResult


VALID_CODE_DRAFT = "def solve():\n    return [4, 5, 6]\n"
VALID_CODE_FINAL = "def solve():\n    return [7, 8, 9]\n"
INVALID_CODE_IMPORT_OS = "import os\ndef solve():\n    return 1\n"
PARENT_CODE = "def solve():\n    return [1, 2, 3]\n"


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


def _context() -> MutationContext:
    return MutationContext(search_mode="exploit", mutation_style="refine")


class StatefulMockClient:
    """Mock LLM client that returns (or raises) queued responses in order.

    ``responses`` may contain:
      - ``str``: returned from the next call
      - ``Exception`` instance: raised on the next call

    Tracks ``call_count`` and records each ``(system_prompt, user_prompt)``
    tuple in ``calls`` for post-hoc assertions.
    """

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses: list[str | Exception] = list(responses)
        self.call_count: int = 0
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.call_count >= len(self._responses):
            raise AssertionError(f"StatefulMockClient exhausted after {self.call_count} calls")
        response = self._responses[self.call_count]
        self.call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


def _run_mutate(mutator: LLMMutator) -> MutationResult:
    return mutator.mutate(
        [_CodeWrapper(PARENT_CODE)],
        "refine",
        _context(),
    )


# ---------------------------------------------------------------------------
# 1. default policy == single_shot
# ---------------------------------------------------------------------------


def test_default_policy_preserves_single_shot_behavior():
    client = StatefulMockClient([VALID_CODE_DRAFT])
    mutator = LLMMutator(client, _domain())
    result = _run_mutate(mutator)

    assert client.call_count == 1
    assert result.success is True
    assert "return [4, 5, 6]" in result.code


# ---------------------------------------------------------------------------
# 2. single_shot metadata unchanged by policy surface
# ---------------------------------------------------------------------------


def test_single_shot_metadata_is_unchanged_by_policy_surface():
    client_default = StatefulMockClient([VALID_CODE_DRAFT])
    mutator_default = LLMMutator(client_default, _domain())
    result_default = _run_mutate(mutator_default)

    client_explicit = StatefulMockClient([VALID_CODE_DRAFT])
    mutator_explicit = LLMMutator(client_explicit, _domain(), mutator_policy="single_shot")
    result_explicit = _run_mutate(mutator_explicit)

    agentic_keys = {
        "mutator_policy",
        "agentic_pass_count",
        "agentic_draft_code",
        "agentic_critique",
        "agentic_fallback",
    }
    historical_keys = {
        "style",
        "targeted_hypotheses",
        "intended_effect",
        "parent_count",
        "mutator_attempts",
    }

    for result in (result_default, result_explicit):
        assert agentic_keys.isdisjoint(result.metadata.keys())
        assert historical_keys.issubset(result.metadata.keys())

    assert set(result_default.metadata.keys()) == set(result_explicit.metadata.keys())


# ---------------------------------------------------------------------------
# 2b. single_shot retry loop preserved after extraction
# ---------------------------------------------------------------------------


def test_single_shot_retry_preserved_after_extraction():
    client = StatefulMockClient(["", VALID_CODE_DRAFT])
    mutator = LLMMutator(client, _domain(), mutator_policy="single_shot")
    result = _run_mutate(mutator)

    assert client.call_count == 2
    assert result.success is True
    assert result.metadata["mutator_attempts"] == 2

    for key in (
        "mutator_policy",
        "agentic_pass_count",
        "agentic_draft_code",
        "agentic_critique",
        "agentic_fallback",
    ):
        assert key not in result.metadata


# ---------------------------------------------------------------------------
# 3. agentic_v1 happy path — 3 LLM calls
# ---------------------------------------------------------------------------


def test_agentic_v1_happy_path_makes_three_calls():
    client = StatefulMockClient(
        [
            VALID_CODE_DRAFT,
            "- bullet one\n- bullet two critique text",
            VALID_CODE_FINAL,
        ]
    )
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert client.call_count == 3
    assert result.success is True
    assert "return [7, 8, 9]" in result.code

    assert result.metadata["mutator_policy"] == "agentic_v1"
    assert result.metadata["agentic_pass_count"] == 3
    assert result.metadata["agentic_fallback"] == "none"
    assert "return [4, 5, 6]" in result.metadata["agentic_draft_code"]
    assert "critique text" in result.metadata["agentic_critique"]


# ---------------------------------------------------------------------------
# 4. agentic_v1 fallback — empty critique
# ---------------------------------------------------------------------------


def test_agentic_v1_falls_back_to_draft_when_critique_is_empty():
    client = StatefulMockClient([VALID_CODE_DRAFT, ""])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert client.call_count == 2
    assert result.success is True
    assert "return [4, 5, 6]" in result.code
    assert result.metadata["agentic_fallback"] == "critique_parse_failed"
    assert result.metadata["agentic_pass_count"] == 2


# ---------------------------------------------------------------------------
# 5. agentic_v1 fallback — finalize parse failure
# ---------------------------------------------------------------------------


def test_agentic_v1_falls_back_to_draft_when_finalize_parse_fails():
    client = StatefulMockClient([VALID_CODE_DRAFT, "critique text", ""])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert client.call_count == 3
    assert result.success is True
    assert "return [4, 5, 6]" in result.code
    assert result.metadata["agentic_fallback"] == "finalize_parse_failed"


# ---------------------------------------------------------------------------
# 6. agentic_v1 fallback — finalize validation failure
# ---------------------------------------------------------------------------


def test_agentic_v1_falls_back_to_draft_when_finalize_validation_fails():
    client = StatefulMockClient([VALID_CODE_DRAFT, "critique text", INVALID_CODE_IMPORT_OS])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert result.success is True
    assert "return [4, 5, 6]" in result.code
    assert result.metadata["agentic_fallback"] == "finalize_validation_failed"


# ---------------------------------------------------------------------------
# 7. agentic_v1 — draft failure returns failure, no retry
# ---------------------------------------------------------------------------


def test_agentic_v1_draft_failure_returns_failure_result_no_retry():
    client = StatefulMockClient([""])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert client.call_count == 1
    assert result.success is False
    assert isinstance(result.errors, list)
    assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# 8. agentic_v1 — draft validation failure returns failure, no retry
# ---------------------------------------------------------------------------


def test_agentic_v1_draft_validation_failure_returns_failure_no_retry():
    client = StatefulMockClient([INVALID_CODE_IMPORT_OS])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    result = _run_mutate(mutator)

    assert client.call_count == 1
    assert result.success is False


# ---------------------------------------------------------------------------
# 9. agentic_v1 — LLMAPIError propagates from draft
# ---------------------------------------------------------------------------


def test_agentic_v1_propagates_llm_api_error_on_draft():
    client = StatefulMockClient([LLMAPIError("synthetic")])
    mutator = LLMMutator(client, _domain(), mutator_policy="agentic_v1")
    with pytest.raises(LLMAPIError, match="synthetic"):
        _run_mutate(mutator)


# ---------------------------------------------------------------------------
# 10. Result schema invariance across policies
# ---------------------------------------------------------------------------


def test_result_schema_is_unchanged_across_policies():
    single_client = StatefulMockClient([VALID_CODE_DRAFT])
    single_mutator = LLMMutator(single_client, _domain(), mutator_policy="single_shot")
    single_result = _run_mutate(single_mutator)

    agentic_client = StatefulMockClient([VALID_CODE_DRAFT, "critique text", VALID_CODE_FINAL])
    agentic_mutator = LLMMutator(agentic_client, _domain(), mutator_policy="agentic_v1")
    agentic_result = _run_mutate(agentic_mutator)

    for result in (single_result, agentic_result):
        assert isinstance(result, MutationResult)
        assert isinstance(result.code, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# 11. Unknown policy raises at construction
# ---------------------------------------------------------------------------


def test_unknown_policy_raises_value_error_at_construction():
    client = StatefulMockClient([])
    with pytest.raises(ValueError) as excinfo:
        LLMMutator(client, _domain(), mutator_policy="xyz")
    message = str(excinfo.value)
    assert "single_shot" in message
    assert "agentic_v1" in message
