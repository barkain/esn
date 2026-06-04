"""Tests for ClaudeAgentMutator (PR A, phase 2.1).

These tests validate the agent-backed mutator end-to-end using a local
``FakeClaudeAgentClient`` (constructor-injected; no ``unittest.mock.patch``,
no real SDK). They cover the four core contracts:

1. A valid agent response produces a ``MutationResult(success=True)`` with
   the expected metadata.
2. A ``ClaudeAgentClientError`` becomes a failed ``MutationResult`` and does
   NOT populate success-only metadata keys.
3. The ``MutatorInputBundle`` handed to the client contains only the 13
   whitelisted fields, hypotheses are scrubbed to ``{id, family, summary}``,
   and no non-whitelisted context sentinel strings leak through.
4. An agent response whose ``code`` is syntactically invalid is rejected by
   the AST validator and the invalid code is NOT smuggled into the result.
"""

from __future__ import annotations

import json

from esn.core.models import EvaluationResult
from esn.engine import ClaudeAgentMutator, MutatorInputBundle
from esn.engine.claude_agent_client import (
    ClaudeAgentClientError,
    ClaudeAgentResponse,
)
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.models import MutationContext


EXPECTED_BUNDLE_KEYS = frozenset(
    {
        "domain_name",
        "domain_description",
        "hard_constraints",
        "allowed_imports",
        "max_code_lines",
        "program_interface",
        "parent_code",
        "style",
        "intended_effect",
        "targeted_hypothesis_ids",
        "top_hypotheses_summary",
        "mutation_style",
        "search_mode",
        "preferred_solution_shape",
        "spectral_guidance",
    }
)


# ---------------------------------------------------------------------------
# Local fake client (constructor-injected; no patch, no real SDK)
# ---------------------------------------------------------------------------


class FakeClaudeAgentClient:
    """Fake ``ClaudeAgentClient`` that returns a canned response or raises.

    Records the last ``MutatorInputBundle`` received on ``last_bundle`` so
    tests can inspect what crossed the isolation boundary.
    """

    def __init__(self, response_or_exception: ClaudeAgentResponse | Exception) -> None:
        self._next = response_or_exception
        self.last_bundle: MutatorInputBundle | None = None

    def run_mutation(self, bundle: MutatorInputBundle) -> ClaudeAgentResponse:
        self.last_bundle = bundle
        if isinstance(self._next, Exception):
            raise self._next
        return self._next


# ---------------------------------------------------------------------------
# ProgramObject doubles
# ---------------------------------------------------------------------------


class _StubProgram:
    """Minimal ProgramObject with a valid ``.code`` attribute."""

    def __init__(self, code: str = "def solve():\n    return 42\n") -> None:
        self.code = code

    def summary(self) -> str:  # pragma: no cover - not called by mutator
        return "stub"

    def structural_hash(self) -> str:  # pragma: no cover - not called by mutator
        return "stub-hash"

    def serialize(self) -> str:  # pragma: no cover - not called by mutator
        return "{}"


class _SummaryForbiddenProgram:
    """ProgramObject that EXPLODES if anything beyond ``.code`` is read.

    Used in the bundle-scrubbing test to prove ``_build_bundle`` never calls
    ``summary()`` / ``structural_hash()`` / ``serialize()``.
    """

    def __init__(self, code: str = "def solve():\n    return 7\n") -> None:
        self.code = code

    def summary(self) -> str:
        raise AssertionError("summary should not be called")

    def structural_hash(self) -> str:
        raise AssertionError("structural_hash should not be called")

    def serialize(self) -> str:
        raise AssertionError("serialize should not be called")


# ---------------------------------------------------------------------------
# Minimal domain + context helpers
# ---------------------------------------------------------------------------


def _make_domain() -> DomainSpec:
    return DomainSpec(
        name="test_domain",
        description="test domain",
        initial_code="def solve():\n    return 0\n",
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=lambda artifact: EvaluationResult(score=0.0, success=True),
        allowed_imports=frozenset({"math"}),
        max_code_lines=50,
        hard_constraints=["no_recursion"],
        program_interface="def solve(): ...",
    )


def _make_simple_context() -> MutationContext:
    """Plain context used by tests that don't care about leakage."""
    return MutationContext(
        search_mode="exploit",
        mutation_style="refine",
        intended_effect="tighten the inner loop",
        targeted_hypothesis_ids=["H1"],
    )


def _make_context_with_sentinels() -> MutationContext:
    """Context with UNIQUE sentinel strings planted in non-whitelisted fields.

    Whitelisted fields (``top_hypotheses``, ``intended_effect``,
    ``targeted_hypothesis_ids``, ``mutation_style``, ``search_mode``) carry
    normal values — but ``top_hypotheses`` entries include extra keys
    (``score``, ``source``) that the scrubber MUST drop.

    Everything else carries a ``CTX_*_SENTINEL`` string so the bundle-JSON
    leak test can grep for them.
    """
    return MutationContext(
        search_mode="exploit",
        mutation_style="refine",
        intended_effect="increase recursion depth",
        targeted_hypothesis_ids=["H1", "H2"],
        top_hypotheses=[
            {
                "id": "H1",
                "family": "F1",
                "summary": "s1",
                "score": 99.9,
                "source": "LEAK-SRC",
            },
            {
                "id": "H2",
                "family": "F2",
                "summary": "s2",
                "score": 42.0,
                "source": "LEAK-SRC",
            },
        ],
        # Non-whitelisted fields — all tagged with CTX_*_SENTINEL strings.
        spectral_guidance={"note": "CTX_SG_SENTINEL"},
        search_temperature=0.0,
        diagnostics={"note": "CTX_DIAG_SENTINEL"},
        score_history={"note": "CTX_SCORE_SENTINEL"},
        error_context="CTX_ERR_SENTINEL",
        best_code="CTX_BEST_CODE_SENTINEL",
        best_score=1234.5,
        recent_attempts=[{"CTX_RECENT_SENTINEL": "leak"}],
        archive_families=["CTX_ARCHIVE_SENTINEL"],
        stagnation_gens=0,
        family_summaries=["CTX_FAM_SUM_SENTINEL"],
        parent_family="CTX_PARENT_FAM_SENTINEL",
        family_failure_reasons={"F1": ["CTX_FFR_SENTINEL"]},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_mutation_result_on_valid_agent_response():
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=3,
        agent_summary="ok",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    assert result.code == "def solve():\n    return 1\n"
    assert result.errors == []
    assert result.metadata["mutator_policy"] == "claude_agent_sdk"
    assert result.metadata["agent_backend"] == "claude_agent_sdk"
    assert result.metadata["agent_turn_count"] == 3
    assert result.metadata["agent_used_research"] is False
    assert result.metadata["parent_count"] == 1


def test_returns_failure_on_client_error():
    fake = FakeClaudeAgentClient(ClaudeAgentClientError("malformed agent response"))
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    assert result.code == ""
    assert len(result.errors) > 0
    joined = " ".join(result.errors).lower()
    assert ("claude agent client error" in joined) or ("malformed" in joined)
    assert result.metadata["mutator_policy"] == "claude_agent_sdk"
    assert result.metadata["parent_count"] == 1
    # Phase-1.2 spec: success-only keys must NOT appear on failure.
    assert "agent_turn_count" not in result.metadata
    assert "agent_summary" not in result.metadata


def test_bundle_contains_only_whitelisted_fields():
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=1,
        agent_summary="ok",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    ctx = _make_context_with_sentinels()
    program = _SummaryForbiddenProgram()

    result = mutator.mutate([program], "refine", ctx)
    # Sanity: the call completed normally (the forbidden methods were not invoked).
    assert result.success is True

    assert fake.last_bundle is not None
    dump = fake.last_bundle.model_dump()

    # Exact-key match against the 13-field whitelist.
    assert set(dump.keys()) == EXPECTED_BUNDLE_KEYS

    # Defense-in-depth: explicitly assert keys a buggy impl might have added.
    # ``spectral_guidance`` is now a whitelisted field (LLMMutator parity);
    # it is NOT in this bad-keys list. Non-whitelisted context state still
    # must not smuggle into the bundle.
    for bad in (
        "parent_summary",
        "diagnostics",
        "score_history",
        "recent_attempts",
        "family_failure_reasons",
        "best_code",
        "best_score",
    ):
        assert bad not in dump

    # No sentinel string leaks through to the serialized bundle.
    # ``CTX_SG_SENTINEL`` is intentionally omitted — ``spectral_guidance`` is
    # a whitelisted prompt-steering channel (mirroring ``LLMMutator``
    # ``src/esn/engine/mutator.py:479-480``), so the context dict's str()
    # representation legitimately flows into the bundle. That case is
    # covered by the dedicated ``test_build_bundle_propagates_...`` tests.
    serialized = json.dumps(dump, default=str)
    for sentinel in (
        "CTX_DIAG_SENTINEL",
        "CTX_RECENT_SENTINEL",
        "CTX_BEST_CODE_SENTINEL",
        "CTX_FFR_SENTINEL",
        "CTX_SCORE_SENTINEL",
        "CTX_ERR_SENTINEL",
        "CTX_ARCHIVE_SENTINEL",
        "CTX_FAM_SUM_SENTINEL",
        "CTX_PARENT_FAM_SENTINEL",
        "LEAK-SRC",
    ):
        assert sentinel not in serialized, f"sentinel {sentinel} leaked into bundle"

    # Hypotheses scrubbed to exactly {id, family, summary}.
    for h in dump["top_hypotheses_summary"]:
        assert set(h.keys()) == {"id", "family", "summary"}


def test_build_bundle_propagates_populated_spectral_guidance():
    """``_build_bundle`` must copy ``context.spectral_guidance`` into the bundle.

    Parity with ``LLMMutator`` at ``src/esn/engine/mutator.py:479-480`` — an
    empty/falsy dict skips the prompt line, a populated dict is rendered
    via its ``str()`` form.
    """
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="ok",
        agent_turn_count=1,
        agent_summary="ok",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    ctx = MutationContext(
        search_mode="exploit",
        mutation_style="refine",
        intended_effect="tighten",
        spectral_guidance={"phase": "exploit", "note": "SPECTRAL_PAYLOAD_XYZ"},
    )

    result = mutator.mutate([_StubProgram()], "refine", ctx)
    assert result.success is True

    assert fake.last_bundle is not None
    # Populated dict -> bundle carries a str representation (parity with
    # ``f"Spectral guidance: {context.spectral_guidance}"``).
    assert fake.last_bundle.spectral_guidance is not None
    assert "SPECTRAL_PAYLOAD_XYZ" in fake.last_bundle.spectral_guidance


def test_build_bundle_spectral_guidance_none_when_context_empty():
    """Empty ``context.spectral_guidance`` dict mirrors LLMMutator gating.

    ``LLMMutator``'s ``if context.spectral_guidance`` skips the line when
    the dict is empty / falsy. The agentic bundle encodes that as ``None``
    so ``_render_prompt`` emits the fallback text.
    """
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="ok",
        agent_turn_count=1,
        agent_summary="ok",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    # Default MutationContext: spectral_guidance defaults to an empty dict.
    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())
    assert result.success is True

    assert fake.last_bundle is not None
    assert fake.last_bundle.spectral_guidance is None


def test_failure_on_ast_validation_error():
    response = ClaudeAgentResponse(
        code="def broken(:",  # syntactically INVALID Python
        diff_summary="x",
        intended_effect="x",
        agent_turn_count=1,
        agent_summary="x",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    # The invalid code must NOT be smuggled into the result.
    assert result.code == ""
    assert len(result.errors) > 0

    joined = " ".join(result.errors).lower()
    assert any(token in joined for token in ("valid", "ast", "syntax", "parse"))
    assert result.metadata["mutator_policy"] == "claude_agent_sdk"


def test_ast_validation_failure_preserves_accounting_metadata():
    """Completed Claude runs are not free just because AST validation fails."""
    response = ClaudeAgentResponse(
        code="def broken(:",  # syntactically INVALID Python
        diff_summary="x",
        intended_effect="x",
        agent_turn_count=4,
        agent_summary="turns=4",
        model="claude-sonnet-4-6",
        input_tokens=123,
        output_tokens=45,
        total_cost_usd=0.01,
        raw_response_text="def broken(:",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    meta = result.metadata
    assert meta["mutator_policy"] == "claude_agent_sdk"
    assert meta["mutation_model"] == "claude-sonnet-4-6"
    assert meta["agent_turn_count"] == 4
    assert meta["input_tokens"] == 123
    assert meta["output_tokens"] == 45
    assert meta["total_cost_usd"] == 0.01
    assert meta["mutator_raw_response"] == "def broken(:"


# ---------------------------------------------------------------------------
# Accounting surface tests (Wave 1)
# ---------------------------------------------------------------------------


def test_mutator_surfaces_accounting_in_metadata():
    """All accounting fields from the response flow into ``metadata``."""
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=3,
        agent_summary="ok",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
        total_cost_usd=0.01,
        raw_response_text="raw agent output",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    meta = result.metadata
    assert meta["mutator_policy"] == "claude_agent_sdk"
    assert meta["agent_backend"] == "claude_agent_sdk"
    assert meta["mutation_model"] == "claude-sonnet-4-6"
    assert meta["agent_turn_count"] == 3
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 200
    assert meta["total_cost_usd"] == 0.01
    assert meta["mutator_raw_response"] == "raw agent output"


def test_mutator_leaves_unavailable_accounting_fields_as_none():
    """If the SDK did not expose tokens/cost/raw text, metadata stores ``None``.

    The mutator must NOT invent data — ``None`` is the honest missing-value
    marker the harness keys off to decide whether accounting is available.
    """
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=2,
        agent_summary="ok",
        model="claude-sonnet-4-6",
        # All four accounting-optional fields left at their ``None`` default.
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    meta = result.metadata
    assert meta["agent_turn_count"] == 2
    assert meta["mutation_model"] == "claude-sonnet-4-6"
    assert meta["input_tokens"] is None
    assert meta["output_tokens"] is None
    assert meta["total_cost_usd"] is None
    assert meta["mutator_raw_response"] is None


def test_mutator_failure_metadata_carries_partial_accounting():
    """Post-run parse failure: metadata carries real accounting from partial.

    The client raised ``ClaudeAgentClientError`` with a ``partial_response``
    attached (SDK completed; parse failed afterwards). The mutator MUST
    thread that partial's accounting fields into the failure metadata so
    the harness does not undercount cost on bad attempts.
    """
    partial = ClaudeAgentResponse(
        code="",
        diff_summary="",
        intended_effect="",
        agent_turn_count=5,
        agent_summary="turns=5",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
        total_cost_usd=0.02,
        raw_response_text="non-json garbage from the SDK",
    )
    fake = FakeClaudeAgentClient(ClaudeAgentClientError("parse failed", partial_response=partial))
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    meta = result.metadata
    # Accounting surface is populated from the partial response.
    assert meta["mutation_model"] == "claude-sonnet-4-6"
    assert meta["agent_turn_count"] == 5
    assert meta["input_tokens"] == 100
    assert meta["output_tokens"] == 200
    assert meta["total_cost_usd"] == 0.02
    assert meta["mutator_raw_response"] == "non-json garbage from the SDK"
    # Base-metadata keys still present.
    assert meta["mutator_policy"] == "claude_agent_sdk"


def test_mutator_failure_metadata_without_partial_response():
    """Mid-stream failure: no accounting keys written (not even as None).

    When ``partial_response`` is absent on the error, the mutator must
    NOT invent accounting keys. The recorder treats missing keys as
    ``None`` via ``.get`` — so the failure is honestly "accounting absent"
    rather than spurious zeros.
    """
    fake = FakeClaudeAgentClient(ClaudeAgentClientError("mid-stream SDK failure"))
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    meta = result.metadata
    # Accounting keys are absent (not None) — matches the pre-existing
    # failure-shape convention the recorder's ``.get`` calls rely on.
    assert "agent_turn_count" not in meta
    assert "input_tokens" not in meta
    assert "output_tokens" not in meta
    assert "mutation_model" not in meta
    assert "mutator_raw_response" not in meta


def test_mutator_preserves_pre_wave1_metadata_keys():
    """Pre-existing keys (``agent_summary``, ``diff_summary``) survive untouched."""
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="pre-wave1-diff",
        intended_effect="return 1",
        agent_turn_count=4,
        agent_summary="pre-wave1-summary",
        model="claude-sonnet-4-6",
        input_tokens=1,
        output_tokens=2,
        total_cost_usd=0.0,
        raw_response_text="raw",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    meta = result.metadata
    # Pre-Wave-1 keys still populated, unchanged semantics.
    assert meta["agent_summary"] == "pre-wave1-summary"
    assert meta["diff_summary"] == "pre-wave1-diff"
    assert meta["agent_turn_count"] == 4
    assert meta["agent_used_research"] is False
    assert meta["parent_count"] == 1


# ---------------------------------------------------------------------------
# research_summary surface — must ALWAYS be present in
# MutationResult.metadata (possibly "") so downstream consumers don't have
# to probe with ``in metadata``. Covers success, AST-validation failure,
# partial-response client error, and pure mid-stream error.
# ---------------------------------------------------------------------------


def test_metadata_carries_research_summary_on_success():
    """Success path propagates ``response.research_summary`` verbatim."""
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=3,
        agent_summary="ok",
        research_summary="adopted convexification heuristic",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    assert result.metadata["research_summary"] == ("adopted convexification heuristic")


def test_metadata_research_summary_defaults_empty_on_legacy_payload():
    """Legacy 3-key payload (no ``research_summary``) lands as ``""``.

    The ``ClaudeAgentResponse`` dataclass defaults ``research_summary`` to
    ``""``; the mutator should not invent anything else.
    """
    response = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="trivial",
        intended_effect="return 1",
        agent_turn_count=2,
        agent_summary="ok",
        # research_summary left at default
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is True
    assert result.metadata["research_summary"] == ""


def test_metadata_research_summary_on_ast_validation_failure():
    """AST-validation failure still surfaces the response's research_summary."""
    response = ClaudeAgentResponse(
        code="def broken(:",
        diff_summary="x",
        intended_effect="x",
        agent_turn_count=1,
        agent_summary="x",
        research_summary="consulted WebSearch on topic X",
    )
    fake = FakeClaudeAgentClient(response)
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    assert result.metadata["research_summary"] == ("consulted WebSearch on topic X")


def test_metadata_research_summary_from_partial_response_on_failure():
    """Client error with a ``partial_response`` threads its research_summary."""
    partial = ClaudeAgentResponse(
        code="",
        diff_summary="",
        intended_effect="",
        agent_turn_count=5,
        agent_summary="turns=5",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
        total_cost_usd=0.02,
        raw_response_text="garbage",
        research_summary="partial research notes",
    )
    fake = FakeClaudeAgentClient(ClaudeAgentClientError("parse failed", partial_response=partial))
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    assert result.metadata["research_summary"] == "partial research notes"


def test_metadata_research_summary_empty_on_pure_failure():
    """Mid-stream failure (no partial): research_summary is still present,
    as ``""`` — callers never have to check ``in metadata``.
    """
    fake = FakeClaudeAgentClient(ClaudeAgentClientError("mid-stream SDK failure"))
    mutator = ClaudeAgentMutator(fake, _make_domain())

    result = mutator.mutate([_StubProgram()], "refine", _make_simple_context())

    assert result.success is False
    assert "research_summary" in result.metadata
    assert result.metadata["research_summary"] == ""


def test_metadata_research_summary_empty_on_empty_parents():
    """Empty-parents guard: research_summary is present as ``""``."""
    mutator = ClaudeAgentMutator(
        FakeClaudeAgentClient(
            ClaudeAgentResponse(
                code="",
                diff_summary="",
                intended_effect="",
                agent_turn_count=0,
                agent_summary="",
            )
        ),
        _make_domain(),
    )

    result = mutator.mutate([], "refine", _make_simple_context())

    assert result.success is False
    assert "research_summary" in result.metadata
    assert result.metadata["research_summary"] == ""
