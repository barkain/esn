# ruff: noqa: S101
"""Tests for the per-call SDK timeout in ``ClaudeAgentSDKClient``.

These tests guard the wall-clock trap added in PR4-B (feat/sdk-call-timeout).
A standalone repro demonstrated that
``claude-sonnet-4-6`` can wedge inside the Claude Agent SDK stream for
300+ seconds without ever emitting a ``ResultMessage``. Without a per-call
timeout, the mutator hangs the entire harness instead of recording a failed
candidate. ``ClaudeAgentSDKClient`` now wraps the FULL ``async for msg in
query(...)`` loop in ``asyncio.wait_for`` and converts a timeout into a
``ClaudeAgentClientError`` (with a best-effort ``partial_response``).

Isolation notes (mirroring ``test_claude_agent_client.py``):

* We never hit the real SDK over the network.
* We patch ``claude_agent_sdk.query`` (the symbol the client lazy-imports)
  with a scripted async generator that either yields immediately, sleeps
  forever, or yields a few partial messages before sleeping.
* We use the real ``AssistantMessage`` / ``ResultMessage`` / ``TextBlock``
  dataclasses where stream-message ``isinstance`` checks need to succeed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest import mock

import pytest

# Optional [agent] extra: these tests lazy-import claude_agent_sdk inside the
# test bodies (mirroring the client). Skip the whole module gracefully when the
# SDK is not installed instead of hard-failing with ModuleNotFoundError.
pytest.importorskip("claude_agent_sdk")

from esn.engine.claude_agent_client import (  # noqa: E402
    ClaudeAgentClientError,
    ClaudeAgentSDKClient,
    MutatorInputBundle,
)


# ---------------------------------------------------------------------------
# Helpers — mirror the existing claude-agent test conventions
# ---------------------------------------------------------------------------


def _make_bundle(parent_code: str = "def solve():\n    return 0\n") -> MutatorInputBundle:
    """Minimal valid bundle (matches ``test_claude_agent_client._make_bundle``)."""
    return MutatorInputBundle(
        domain_name="test_domain",
        domain_description="test domain",
        hard_constraints=[],
        allowed_imports=[],
        max_code_lines=50,
        program_interface="def solve(): ...",
        parent_code=parent_code,
        style="refine",
        intended_effect="",
        targeted_hypothesis_ids=[],
        top_hypotheses_summary=[],
        mutation_style="refine",
        search_mode="exploit",
    )


_VALID_AGENT_BODY = (
    '```json\n{"code": "def solve():\\n    return 1\\n", '
    '"diff_summary": "ok", "intended_effect": "return 1", '
    '"research_summary": ""}\n```'
)


def _make_quick_query():
    """Fake ``query`` that yields one Assistant + one Result and exits.

    Uses the real SDK dataclasses so the ``isinstance`` checks in
    ``_run_inner`` match.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-haiku-4-5-20251001",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sess-quick",
        total_cost_usd=0.001,
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    async def _agen(*_args: Any, **_kwargs: Any):
        yield assistant
        yield result

    def _query(*args: Any, **kwargs: Any):
        return _agen(*args, **kwargs)

    return _query


def _make_hung_query(*, partial_messages: list[Any] | None = None, sleep_s: float = 30.0):
    """Fake ``query`` that yields optional partials, then sleeps forever.

    Consumed inside ``asyncio.wait_for(...)``; the sleep models the SDK
    child wedging without a ``ResultMessage`` (the failure mode the
    timeout guards against).
    """
    msgs = list(partial_messages or [])

    async def _agen(*_args: Any, **_kwargs: Any):
        for m in msgs:
            yield m
        # Sleep long past any reasonable test timeout. ``asyncio.wait_for``
        # cancels the task so the sleep never actually completes.
        await asyncio.sleep(sleep_s)
        # Unreachable: sentinel only, in case a caller forgets to wrap
        # in ``wait_for``.
        yield {"unreachable": True}

    def _query(*args: Any, **kwargs: Any):
        return _agen(*args, **kwargs)

    return _query


# ---------------------------------------------------------------------------
# 1. Constructor surface
# ---------------------------------------------------------------------------


def test_default_timeout_is_300_seconds() -> None:
    """The default timeout must match the documented 5-minute conservative bound."""
    client = ClaudeAgentSDKClient()
    assert client.call_timeout_seconds == 300.0


def test_constructor_accepts_custom_timeout() -> None:
    """Custom ``call_timeout_seconds`` round-trips through the constructor as a float."""
    client = ClaudeAgentSDKClient(call_timeout_seconds=10.5)
    assert client.call_timeout_seconds == pytest.approx(10.5)


def test_constructor_coerces_int_timeout_to_float() -> None:
    """Int-valued timeouts are accepted (CLI may pass int) and stored as float."""
    client = ClaudeAgentSDKClient(call_timeout_seconds=120)
    assert isinstance(client.call_timeout_seconds, float)
    assert client.call_timeout_seconds == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# 2. Timeout behavior
# ---------------------------------------------------------------------------


def test_timeout_raises_client_error_on_hang() -> None:
    """A hung SDK stream must be trapped as ``ClaudeAgentClientError``.

    The fake query sleeps far longer than the configured timeout. The
    client must raise within roughly the timeout duration (NOT the sleep
    duration), and the error message must mention the timeout.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    client = ClaudeAgentSDKClient(
        model="claude-haiku-4-5-20251001",
        call_timeout_seconds=1.0,
    )
    bundle = _make_bundle()

    fake_query = _make_hung_query(sleep_s=30.0)
    start = time.monotonic()
    with mock.patch.object(sdk, "query", fake_query):
        with pytest.raises(ClaudeAgentClientError, match="timed out"):
            client.run_mutation(bundle)
    elapsed = time.monotonic() - start
    # Tolerance: ~0.5s of asyncio scheduling + tempdir setup overhead is
    # generous; the sleep itself was 30s, so anything under ~3s is well
    # short of the unconfigured behavior.
    assert elapsed < 3.0, f"timeout did not fire within ~1s + slack; took {elapsed:.2f}s"


def test_no_timeout_when_response_arrives_quickly() -> None:
    """A well-behaved stream must complete without raising the timeout error.

    ``call_timeout_seconds`` is generous (10s) and the fake query yields
    a complete Assistant + Result pair immediately. The call must return
    a parsed ``ClaudeAgentResponse`` with no errors raised.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    client = ClaudeAgentSDKClient(
        model="claude-haiku-4-5-20251001",
        call_timeout_seconds=10.0,
    )
    bundle = _make_bundle()

    fake_query = _make_quick_query()
    with mock.patch.object(sdk, "query", fake_query):
        response = client.run_mutation(bundle)

    # Sanity: the parsed response carries the expected accounting from
    # our scripted ResultMessage.
    assert response.code.startswith("def solve()")
    assert response.agent_turn_count == 1
    assert response.input_tokens == 10
    assert response.output_tokens == 20


def test_partial_transcript_preserved_on_timeout() -> None:
    """On timeout, the partial response must reflect text seen before the wedge.

    The fake query yields two ``AssistantMessage`` text blocks containing
    distinctive substrings, then sleeps far past the timeout. After the
    ``ClaudeAgentClientError`` fires, the attached ``partial_response``
    must contain the concatenated text in ``raw_response_text`` — that
    is the surface the mutator's failed-candidate metadata reads.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    msg_a = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="partial-fragment-A")],
        model="claude-haiku-4-5-20251001",
    )
    msg_b = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="partial-fragment-B")],
        model="claude-haiku-4-5-20251001",
    )

    client = ClaudeAgentSDKClient(
        model="claude-haiku-4-5-20251001",
        call_timeout_seconds=1.0,
    )
    bundle = _make_bundle()

    fake_query = _make_hung_query(
        partial_messages=[msg_a, msg_b],
        sleep_s=30.0,
    )
    with mock.patch.object(sdk, "query", fake_query):
        with pytest.raises(ClaudeAgentClientError) as excinfo:
            client.run_mutation(bundle)

    err = excinfo.value
    # Best-effort partial response is attached so the mutator can record
    # real (partial) accounting on the timeout path.
    assert err.partial_response is not None
    raw = err.partial_response.raw_response_text or ""
    assert "partial-fragment-A" in raw
    assert "partial-fragment-B" in raw
    # Model identity must still be stamped on the partial response so
    # downstream metadata writes carry the configured model id.
    assert err.partial_response.model == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# 3. Mutator wiring — the timeout error must carry the existing partial-
#    response contract so the mutator's catch path attaches accounting.
# ---------------------------------------------------------------------------


def test_timeout_error_partial_response_carries_accounting_surface() -> None:
    """The timeout's ``partial_response`` must expose all accounting fields.

    ``ClaudeAgentMutator``'s catch branch (see
    ``src/esn/engine/claude_agent_mutator.py:154``) reads
    ``partial.model``, ``partial.agent_turn_count``, ``partial.input_tokens``,
    ``partial.output_tokens``, ``partial.total_cost_usd``,
    ``partial.raw_response_text``, and ``partial.research_summary`` to
    populate ``MutationResult.metadata``. This test guards that the
    timeout-path partial response carries every one of those attributes
    (with safe ``None`` / empty defaults when the SDK never produced a
    ``ResultMessage``).
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    client = ClaudeAgentSDKClient(
        model="claude-haiku-4-5-20251001",
        call_timeout_seconds=0.5,
    )
    bundle = _make_bundle()

    fake_query = _make_hung_query(sleep_s=30.0)
    with mock.patch.object(sdk, "query", fake_query):
        with pytest.raises(ClaudeAgentClientError) as excinfo:
            client.run_mutation(bundle)

    partial = excinfo.value.partial_response
    assert partial is not None
    # Required attributes the mutator reads (must exist; values may be
    # None / 0 / "" because no ResultMessage ever arrived).
    assert partial.model == "claude-haiku-4-5-20251001"
    assert partial.agent_turn_count == 0
    assert partial.input_tokens is None
    assert partial.output_tokens is None
    assert partial.total_cost_usd is None
    assert partial.research_summary == ""
    # ``raw_response_text`` is None when no Assistant text arrived
    # (this hung-query has no partials). The earlier
    # ``test_partial_transcript_preserved_on_timeout`` exercises the
    # populated case.
    assert partial.raw_response_text is None
