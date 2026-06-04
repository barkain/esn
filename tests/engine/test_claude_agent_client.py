# ruff: noqa: S101
"""Tests for ``ClaudeAgentSDKClient`` accounting surface (Wave 1).

These tests exercise the SDK adapter's handling of accounting fields
(``ResultMessage.num_turns`` / ``usage`` / ``total_cost_usd``, concatenated
``AssistantMessage`` text, and the configured model) WITHOUT calling the real
``claude_agent_sdk.query`` — they swap the symbols the lazy-import pulls in
with scripted async generators.

Isolation notes:

* We never hit the real SDK over the network.
* We patch ``claude_agent_sdk.query`` (the name the client lazy-imports) so
  ``asyncio.run(_run())`` consumes our scripted messages.
* We use the real ``AssistantMessage`` / ``ResultMessage`` / ``TextBlock``
  dataclasses so the ``isinstance`` checks inside ``_run`` match.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

# Optional [agent] extra: these tests lazy-import claude_agent_sdk inside the
# test bodies (mirroring the client). Skip the whole module gracefully when the
# SDK is not installed instead of hard-failing with ModuleNotFoundError.
pytest.importorskip("claude_agent_sdk")

from esn.engine.claude_agent_client import (  # noqa: E402
    ClaudeAgentClientError,
    ClaudeAgentResponse,
    ClaudeAgentSDKClient,
    MutatorInputBundle,
    _build_sdk_child_env,
    _parse_agent_output,
    _render_prompt,
)


# ---------------------------------------------------------------------------
# Helpers: build a valid bundle + a scripted-message async generator
# ---------------------------------------------------------------------------


def _make_bundle(parent_code: str = "def solve():\n    return 0\n") -> MutatorInputBundle:
    """Minimal valid bundle — fields match ``MutatorInputBundle``'s whitelist."""
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
    '"diff_summary": "ok", "intended_effect": "return 1"}\n```'
)


def _make_scripted_query(messages: list[Any]):
    """Return a fake ``query(prompt, options)`` that yields ``messages``.

    ``query`` in the real SDK is an async generator callable; we build a
    matching shape (regular function returning an async iterator) so the
    ``async for message in query(...)`` loop inside ``_run`` is happy.
    """

    async def _agen(*_args: Any, **_kwargs: Any):
        for msg in messages:
            yield msg

    def _query(*args: Any, **kwargs: Any):
        return _agen(*args, **kwargs)

    return _query


# ---------------------------------------------------------------------------
# 1. Response dataclass carries the new accounting fields
# ---------------------------------------------------------------------------


def test_render_prompt_includes_wallclock_runtime_hint() -> None:
    """``_render_prompt`` must tell the agent to install an elapsed-time guard
    when the bundle's allowlist permits ``time``.

    Same intent as the ``LLMMutator`` system-prompt test over in
    ``test_llm_components.py`` — mirrored here because the Claude-agent
    backend constructs its prompt from ``MutatorInputBundle`` via
    ``_render_prompt`` and NOT via ``LLMMutator._build_system_prompt``.
    Both surfaces must carry the bounded-runtime hint, and both must
    render it domain-aware. Empty ``allowed_imports`` = unrestricted
    (consistent with the rendered prompt block), so ``time`` is allowed.
    """
    # Branch 1: empty allowlist = unrestricted -> concrete time.time().
    bundle = _make_bundle()
    prompt = _render_prompt(bundle)
    assert "time.time()" in prompt, (
        "rendered agent prompt (unrestricted allowlist) is missing the "
        f"concrete 'time.time()' hint. Got:\n{prompt}"
    )
    assert "wall-clock" in prompt.lower()

    # Branch 2: allowlist names ``time`` -> concrete time.time().
    bundle_with_time = bundle.model_copy(update={"allowed_imports": ["time", "math"]})
    prompt_with_time = _render_prompt(bundle_with_time)
    assert "time.time()" in prompt_with_time


def test_render_prompt_uses_generic_runtime_hint_when_time_disallowed() -> None:
    """When the bundle's allowlist is non-empty but excludes ``time``, the
    rendered prompt must NOT name ``time.time()`` — otherwise the post-
    mutation AST validator in that domain would reject what the hint
    induced. The generic iteration-cap variant is used instead.
    """
    bundle = _make_bundle().model_copy(update={"allowed_imports": ["numpy", "math"]})
    prompt = _render_prompt(bundle)
    assert "time.time" not in prompt, (
        f"rendered agent prompt (time-disallowed bundle) must NOT name time.time(). Got:\n{prompt}"
    )
    assert "iteration cap" in prompt.lower(), (
        "rendered agent prompt (time-disallowed bundle) must carry the "
        f"generic iteration-cap variant. Got:\n{prompt}"
    )


# ---------------------------------------------------------------------------
# Spectral-guidance parity with ``LLMMutator`` (``src/esn/engine/mutator.py:479-480``)
# ---------------------------------------------------------------------------


def test_mutator_input_bundle_default_spectral_guidance_is_none() -> None:
    """The new field defaults to ``None`` so existing call-sites are unaffected."""
    bundle = _make_bundle()
    assert bundle.spectral_guidance is None


def test_mutator_input_bundle_accepts_explicit_spectral_guidance() -> None:
    """The frozen / ``extra='forbid'`` bundle accepts the new field when set."""
    bundle = _make_bundle().model_copy(
        update={"spectral_guidance": "explore lower-frequency structural modes"}
    )
    assert bundle.spectral_guidance == "explore lower-frequency structural modes"


def test_render_prompt_includes_populated_spectral_guidance() -> None:
    """When populated, the prompt renders ``# Spectral guidance`` + the text.

    Mirrors ``test_preferred_solution_shape`` parity style: populated case
    must NOT render the fallback.
    """
    text = "emphasise combinatorial swaps over continuous adjustments"
    bundle = _make_bundle().model_copy(update={"spectral_guidance": text})
    prompt = _render_prompt(bundle)
    assert "# Spectral guidance" in prompt
    assert text in prompt
    assert "(no spectral guidance provided)" not in prompt


def test_render_prompt_spectral_guidance_fallback_when_none() -> None:
    """Without guidance, the prompt emits the explicit fallback wording.

    The fallback keeps the section present (so prompt structure is stable)
    and states the absence plainly.
    """
    bundle = _make_bundle()  # spectral_guidance defaults to None
    prompt = _render_prompt(bundle)
    assert "# Spectral guidance" in prompt
    assert "(no spectral guidance provided)" in prompt


def test_render_prompt_spectral_guidance_fallback_in_research_mode() -> None:
    """The section is rendered regardless of the tool-use mode.

    Both no-tools and research-enabled prompts route through the same
    ``_render_prompt`` body below the protocol block.
    """
    bundle = _make_bundle()
    prompt = _render_prompt(bundle, mutator_tools="research")
    assert "# Spectral guidance" in prompt
    assert "(no spectral guidance provided)" in prompt


def test_claude_agent_response_carries_accounting_fields() -> None:
    resp = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="return 1",
        agent_turn_count=3,
        agent_summary="ok",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=200,
        total_cost_usd=0.01,
        raw_response_text="hello world",
    )

    assert resp.model == "claude-sonnet-4-6"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 200
    assert resp.total_cost_usd == pytest.approx(0.01)
    assert resp.raw_response_text == "hello world"


def test_claude_agent_response_accounting_defaults_are_none() -> None:
    """Missing accounting fields default to safe values (not invented data)."""
    resp = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="return 1",
        agent_turn_count=1,
        agent_summary="ok",
    )

    assert resp.input_tokens is None
    assert resp.output_tokens is None
    assert resp.total_cost_usd is None
    assert resp.raw_response_text is None
    # ``model`` defaults to empty string rather than None so it's type-safe
    # as a pass-through to harness string fields; absence is observable via
    # truthiness.
    assert resp.model == ""


# ---------------------------------------------------------------------------
# 2. Public model attribute on the SDK client
# ---------------------------------------------------------------------------


def test_sdk_client_exposes_configured_model() -> None:
    client = ClaudeAgentSDKClient(model="claude-opus-4-x")
    assert client.model == "claude-opus-4-x"


def test_sdk_client_exposes_default_model() -> None:
    client = ClaudeAgentSDKClient()
    # The recon fixed the default at ``claude-haiku-4-5-20251001``; if the default
    # moves, update this assertion deliberately.
    assert client.model == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# 3. End-to-end: accounting flows through ``run_mutation``
# ---------------------------------------------------------------------------


def _run_with_scripted_messages(
    messages: list[Any],
    *,
    model: str = "claude-sonnet-4-6",
) -> ClaudeAgentResponse:
    """Execute ``ClaudeAgentSDKClient.run_mutation`` against a scripted stream.

    We import the real SDK names once, then patch ``claude_agent_sdk.query``
    with our fake while letting ``isinstance`` checks use the real classes.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415 — mirror client's lazy import

    client = ClaudeAgentSDKClient(model=model)
    bundle = _make_bundle()

    fake_query = _make_scripted_query(messages)
    with mock.patch.object(sdk, "query", fake_query):
        return client.run_mutation(bundle)


def test_sdk_client_populates_accounting_from_result_message() -> None:
    import claude_agent_sdk as sdk  # noqa: PLC0415

    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1234,
        duration_api_ms=1000,
        is_error=False,
        num_turns=3,
        session_id="sess-1",
        total_cost_usd=0.01,
        usage={"input_tokens": 100, "output_tokens": 200},
    )

    response = _run_with_scripted_messages([assistant, result], model="claude-sonnet-4-6")

    assert response.agent_turn_count == 3
    assert response.input_tokens == 100
    assert response.output_tokens == 200
    assert response.total_cost_usd == pytest.approx(0.01)
    assert response.model == "claude-sonnet-4-6"
    assert response.code.startswith("def solve()")


def test_sdk_client_leaves_token_fields_none_when_usage_absent() -> None:
    import claude_agent_sdk as sdk  # noqa: PLC0415

    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="sess-2",
        total_cost_usd=None,
        usage=None,
    )

    response = _run_with_scripted_messages([assistant, result])

    assert response.agent_turn_count == 2
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.total_cost_usd is None


def test_sdk_client_accepts_alt_usage_key_names() -> None:
    """``prompt_tokens`` / ``completion_tokens`` fallback is honored."""
    import claude_agent_sdk as sdk  # noqa: PLC0415

    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sess-3",
        usage={"prompt_tokens": 50, "completion_tokens": 75},
    )

    response = _run_with_scripted_messages([assistant, result])

    assert response.input_tokens == 50
    assert response.output_tokens == 75


def test_sdk_client_captures_raw_response_text() -> None:
    """Concatenated assistant text blocks become ``raw_response_text``.

    We yield TWO TextBlocks across two AssistantMessages and verify the
    client preserves the newline-joined text that the existing ``_run`` loop
    builds (matching the pre-existing concatenation semantics).
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    # First text block is a non-JSON header; second is the fenced JSON the
    # parser consumes. ``raw_response_text`` should contain BOTH joined.
    first = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="hello")],
        model="claude-sonnet-4-6",
    )
    second = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="sess-4",
    )

    response = _run_with_scripted_messages([first, second, result])

    assert response.raw_response_text is not None
    assert response.raw_response_text.startswith("hello\n")
    assert _VALID_AGENT_BODY in response.raw_response_text


# ---------------------------------------------------------------------------
# 4. Partial-response accounting on post-run parse failure (PR #29 amend)
# ---------------------------------------------------------------------------


def test_client_error_carries_partial_response_on_parse_failure() -> None:
    """Parse-after-run failure attaches real accounting to the raised error.

    The SDK ``_run()`` completed with ``num_turns=5`` and usage; the
    assistant text is non-JSON garbage the parser rejects. The mutator
    needs the accounting so a failed-parse attempt is not free.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    # Non-JSON, non-fenced text — ``_parse_agent_output`` will raise.
    garbage = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="this is not json and has no fence")],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=5,
        session_id="sess-parse-fail",
        total_cost_usd=0.02,
        usage={"input_tokens": 100, "output_tokens": 200},
    )

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    bundle = _make_bundle()

    fake_query = _make_scripted_query([garbage, result])
    with mock.patch.object(sdk, "query", fake_query):
        with pytest.raises(ClaudeAgentClientError) as excinfo:
            client.run_mutation(bundle)

    partial = excinfo.value.partial_response
    assert partial is not None
    assert partial.agent_turn_count == 5
    assert partial.input_tokens == 100
    assert partial.output_tokens == 200
    assert partial.total_cost_usd == pytest.approx(0.02)
    assert partial.model == "claude-sonnet-4-6"
    # Raw text is preserved even though parsing failed.
    assert partial.raw_response_text is not None
    assert "not json" in partial.raw_response_text
    # Content fields are empty — we never invent data.
    assert partial.code == ""
    assert partial.diff_summary == ""
    assert partial.intended_effect == ""


def test_client_error_carries_no_partial_response_on_mid_stream_failure() -> None:
    """Mid-stream SDK failure leaves ``partial_response`` as ``None``.

    If ``_run()`` itself raises (network/SDK error before a ResultMessage),
    accounting is genuinely absent and we must NOT fabricate a partial.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    bundle = _make_bundle()

    def _exploding_query(*_args: Any, **_kwargs: Any):
        async def _agen():
            # Raise before yielding anything — simulates a mid-stream fault.
            raise RuntimeError("simulated network failure")
            yield  # pragma: no cover — unreachable, makes this an async gen

        return _agen()

    with mock.patch.object(sdk, "query", _exploding_query):
        with pytest.raises(ClaudeAgentClientError) as excinfo:
            client.run_mutation(bundle)

    assert excinfo.value.partial_response is None


# ---------------------------------------------------------------------------
# 5. SDK child env sanitization (PR E: Claude subscription auth)
# ---------------------------------------------------------------------------


def test_sdk_prompt_is_streaming_not_string() -> None:
    """When ``can_use_tool`` is set, prompt must be an ``AsyncIterable[dict]``.

    The SDK (claude_agent_sdk 0.1.63) raises
    ``ValueError("can_use_tool callback requires streaming mode. "
                 "Please provide prompt as an AsyncIterable instead of a string.")``
    when ``options.can_use_tool`` is set and ``prompt`` is a ``str`` — see
    ``claude_agent_sdk/_internal/client.py`` ``process_query`` (line ~57).

    This test pins that ``ClaudeAgentSDKClient`` always wraps its prompt as
    an async iterable, so the deny-all Layer 3 isolation callback can remain
    installed.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415 — mirror client's lazy import

    captured: dict[str, Any] = {}

    def fake_query(*, prompt: Any, options: Any) -> Any:
        captured["prompt"] = prompt
        captured["options"] = options

        async def _agen() -> Any:
            # Minimal successful stream: one assistant text + a terminal result.
            yield sdk.AssistantMessage(
                content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
                model="claude-sonnet-4-6",
            )
            yield sdk.ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-streaming",
            )

        return _agen()

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    bundle = _make_bundle()

    with mock.patch.object(sdk, "query", fake_query):
        response = client.run_mutation(bundle)

    # Sanity: the scripted stream was consumed end-to-end.
    assert response.agent_turn_count == 1

    prompt = captured["prompt"]
    # The core pins: prompt must not be a plain string and must be an
    # async iterable so the SDK recognises streaming mode.
    assert not isinstance(prompt, str), f"prompt must not be a str; got {type(prompt)}"
    assert hasattr(prompt, "__aiter__"), (
        f"prompt must be an AsyncIterable (expose __aiter__); got {type(prompt)}"
    )

    # Belt-and-suspenders: the deny-all can_use_tool callback is still installed.
    assert captured["options"].can_use_tool is not None


def test_sdk_child_env_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper strips Claude Code session markers and sets the skip flag.

    When this process is itself a Claude Code CLI session, the SDK's spawned
    child CLI refuses to start if it inherits ``CLAUDECODE`` /
    ``CLAUDE_CODE_ENTRYPOINT``. The helper must drop both and set
    ``CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1``, while preserving unrelated
    environment variables.
    """
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("SOME_OTHER_VAR", "preserved")

    env = _build_sdk_child_env()

    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK") == "1"
    assert env.get("SOME_OTHER_VAR") == "preserved"


# ---------------------------------------------------------------------------
# 6. --mutator-tools {none,research} tool-policy resolution
# ---------------------------------------------------------------------------


def test_mutator_tools_default_none_resolves_to_empty_list() -> None:
    """Default: ``mutator_tools='none'`` -> ``allowed_tools=[]``.

    The constructor resolves the mode at init time so bad values fail
    immediately (no network call needed).
    """
    client = ClaudeAgentSDKClient()
    assert client.mutator_tools == "none"
    assert client._allowed_tools == []  # noqa: SLF001


def test_mutator_tools_explicit_none_resolves_to_empty_list() -> None:
    client = ClaudeAgentSDKClient(mutator_tools="none")
    assert client.mutator_tools == "none"
    assert client._allowed_tools == []  # noqa: SLF001


def test_mutator_tools_research_resolves_to_websearch_webfetch_only() -> None:
    """``mutator_tools='research'`` -> exactly WebSearch + WebFetch today.

    NO evaluator tools (Bash / Read / Write / Edit) may appear. This is
    the central no-eval boundary assertion.
    """
    client = ClaudeAgentSDKClient(mutator_tools="research")
    assert client.mutator_tools == "research"
    assert client._allowed_tools == ["WebSearch", "WebFetch"]  # noqa: SLF001
    # Explicit no-eval allowlist subset check (mirrors the runtime
    # guard inside ``_resolve_tools``).
    tools = client._allowed_tools  # noqa: SLF001
    assert "Bash" not in tools
    assert "Read" not in tools
    assert "Write" not in tools
    assert "Edit" not in tools


def test_mutator_tools_legacy_web_value_rejected() -> None:
    """The old ``web`` value must now fail loudly.

    Hard rename with
    no alias. A leftover ``--mutator-tools web`` invocation must SystemExit
    at argparse and, past argparse, raise at the SDK client boundary.
    """
    with pytest.raises(ValueError, match="mutator_tools must be"):
        ClaudeAgentSDKClient(mutator_tools="web")


def test_mutator_tools_invalid_value_raises_value_error() -> None:
    """Anything other than 'none' / 'research' must raise at construction.

    Fail fast: we don't want a typo to silently downgrade to 'none'.
    """
    with pytest.raises(ValueError, match="mutator_tools must be"):
        ClaudeAgentSDKClient(mutator_tools="everything")
    with pytest.raises(ValueError, match="mutator_tools must be"):
        ClaudeAgentSDKClient(mutator_tools="")
    with pytest.raises(ValueError, match="mutator_tools must be"):
        ClaudeAgentSDKClient(mutator_tools="bash")


def test_resolve_tools_classmethod_allowlist_guard() -> None:
    """The ``_resolve_tools`` classmethod enforces the research subset.

    If a future edit to ``_RESEARCH_TOOLS`` accidentally adds ``Bash`` but
    leaves the resolver alone, the subset check would still hold (since
    the set is drawn from ``_RESEARCH_TOOLS`` itself). The stronger guard
    is that the allowlist CONTENT is fixed, so we pin it directly here.
    """
    assert ClaudeAgentSDKClient._RESEARCH_TOOLS == ("WebSearch", "WebFetch")  # noqa: SLF001
    # And the resolver produces exactly that, in the declared order.
    assert (
        ClaudeAgentSDKClient._resolve_tools("research")  # noqa: SLF001
        == ["WebSearch", "WebFetch"]
    )
    assert ClaudeAgentSDKClient._resolve_tools("none") == []  # noqa: SLF001


def test_resolve_tools_rejects_legacy_web_alias() -> None:
    """The classmethod path also rejects the old ``web`` value (no alias)."""
    with pytest.raises(ValueError, match="mutator_tools must be"):
        ClaudeAgentSDKClient._resolve_tools("web")  # noqa: SLF001


def test_mutator_tools_research_passes_correct_allowed_tools_to_sdk() -> None:
    """End-to-end: research mode threads the research tools into ``ClaudeAgentOptions``.

    Patches the SDK's ``query`` to capture the options object so we can
    pin ``allowed_tools`` without hitting the network. The deny-all
    callback is replaced by the research-gate callback, which we also verify
    is present.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def fake_query(*, prompt: Any, options: Any) -> Any:
        captured["options"] = options

        async def _agen() -> Any:
            yield sdk.AssistantMessage(
                content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
                model="claude-sonnet-4-6",
            )
            yield sdk.ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-research",
            )

        return _agen()

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6", mutator_tools="research")
    bundle = _make_bundle()
    with mock.patch.object(sdk, "query", fake_query):
        client.run_mutation(bundle)

    options = captured["options"]
    # These are the two pins this test is paid to enforce.
    assert options.allowed_tools == ["WebSearch", "WebFetch"]
    # The belt-and-suspenders callback is still installed (the gate
    # version now permits WebSearch/WebFetch but still denies evaluator
    # tools).
    assert options.can_use_tool is not None


def test_mutator_tools_none_passes_empty_allowed_tools_to_sdk() -> None:
    """Regression: ``none`` mode preserves today's ``allowed_tools=[]``."""
    import claude_agent_sdk as sdk  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def fake_query(*, prompt: Any, options: Any) -> Any:
        captured["options"] = options

        async def _agen() -> Any:
            yield sdk.AssistantMessage(
                content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
                model="claude-sonnet-4-6",
            )
            yield sdk.ResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-none",
            )

        return _agen()

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    bundle = _make_bundle()
    with mock.patch.object(sdk, "query", fake_query):
        client.run_mutation(bundle)

    assert captured["options"].allowed_tools == []


# ---------------------------------------------------------------------------
# 7. Proposer transcript plumbing (feature/mutator-trace-persistence)
# ---------------------------------------------------------------------------


def test_sdk_client_writes_proposer_transcript_when_context_set(
    tmp_path: Any,
) -> None:
    """End-to-end: set_transcript_context -> run_mutation -> file on disk.

    Drives a scripted stream containing a ``ToolUseBlock`` (WebSearch) and
    a ``ToolResultBlock`` via a ``UserMessage`` — the same shape the real
    SDK emits. Verifies the transcript filename + key section headers.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    tool_use = sdk.ToolUseBlock(
        id="toolu_1",
        name="WebSearch",
        input={"query": "Strassen matrix multiplication"},
    )
    assistant_with_tool = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="Let me search."), tool_use],
        model="claude-sonnet-4-6",
    )
    tool_result = sdk.ToolResultBlock(
        tool_use_id="toolu_1",
        content=[
            {
                "title": "Strassen algorithm",
                "url": "https://en.wikipedia.org/wiki/Strassen_algorithm",
                "snippet": "A divide-and-conquer algorithm for matrix multiplication.",
            }
        ],
        is_error=False,
    )
    user_with_result = sdk.UserMessage(
        content=[tool_result],
    )
    assistant_final = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="sess-transcript",
    )

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6", mutator_tools="research")
    client.set_transcript_context(run_dir=tmp_path, generation=5, variant_id="v1")
    bundle = _make_bundle()

    fake_query = _make_scripted_query(
        [assistant_with_tool, user_with_result, assistant_final, result]
    )
    with mock.patch.object(sdk, "query", fake_query):
        response = client.run_mutation(bundle)

    # Parsed program is unchanged by the transcript side effect.
    assert response.code.startswith("def solve()")
    assert response.agent_turn_count == 2

    transcript_path = tmp_path / "transcripts" / "gen_05_tv1_proposer.md"
    assert transcript_path.exists()
    body = transcript_path.read_text()
    assert "# Proposer turn — gen 05 / variant v1" in body
    assert "**Mutator mode**: agentic mutation (research-enabled)" in body
    assert "**Backend**: claude_agent_sdk" in body
    assert "**Model**: claude-sonnet-4-6" in body
    assert "**Tool invocations**: 1 WebSearch" in body
    assert "### Tool use #1 — WebSearch" in body
    assert '**Query**: "Strassen matrix multiplication"' in body
    assert "### Tool result #1" in body
    assert "## Final response (parsed into program)" in body
    # Verbatim final text.
    assert _VALID_AGENT_BODY in body


def test_sdk_client_no_transcript_when_context_unset(
    tmp_path: Any,
) -> None:
    """Without ``set_transcript_context``, no file is written.

    The ``transcripts/`` dir must not even be created. This preserves the
    contract for unit tests and for callers that don't want the side
    effect.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=_VALID_AGENT_BODY)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sess-no-transcript",
    )

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    bundle = _make_bundle()

    fake_query = _make_scripted_query([assistant, result])
    with mock.patch.object(sdk, "query", fake_query):
        client.run_mutation(bundle)

    # No transcript directory created on this path.
    assert not (tmp_path / "transcripts").exists()


def test_sdk_client_transcript_written_on_parse_failure(
    tmp_path: Any,
) -> None:
    """A post-run parse failure still writes a transcript.

    Parse failures are the MOST useful attempt to inspect — the raw final
    text + any tool-use evidence must land on disk even though
    ``run_mutation`` raises.
    """
    import claude_agent_sdk as sdk  # noqa: PLC0415

    garbage = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="not json, no fence")],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="sess-parse-fail-transcript",
    )

    client = ClaudeAgentSDKClient(model="claude-sonnet-4-6")
    client.set_transcript_context(run_dir=tmp_path, generation=2, variant_id="vpf")
    bundle = _make_bundle()

    fake_query = _make_scripted_query([garbage, result])
    with mock.patch.object(sdk, "query", fake_query):
        import pytest as _pytest  # local alias to avoid shadowing at top

        with _pytest.raises(ClaudeAgentClientError):
            client.run_mutation(bundle)

    transcript = tmp_path / "transcripts" / "gen_02_tvpf_proposer.md"
    assert transcript.exists()
    body = transcript.read_text()
    assert "not json, no fence" in body
    assert "## Final response (parsed into program)" in body


# ---------------------------------------------------------------------------
# 8. research_summary field in ClaudeAgentResponse + parser tolerance
# ---------------------------------------------------------------------------


def test_claude_agent_response_research_summary_defaults_to_empty_string() -> None:
    """``research_summary`` is a required field with a safe default.

    Empty string
    when no research was used, otherwise a short (<= 300 chars) note.
    The dataclass default is ``""`` so no-tools / legacy callers are
    tolerated without an explicit kwarg.
    """
    resp = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="return 1",
        agent_turn_count=1,
        agent_summary="ok",
    )
    assert resp.research_summary == ""


def test_claude_agent_response_carries_research_summary_when_set() -> None:
    resp = ClaudeAgentResponse(
        code="def solve():\n    return 1\n",
        diff_summary="ok",
        intended_effect="return 1",
        agent_turn_count=2,
        agent_summary="ok",
        research_summary="Adopted Strassen 2-level recursion.",
    )
    assert resp.research_summary == "Adopted Strassen 2-level recursion."


def test_parse_agent_output_captures_research_summary_when_present() -> None:
    """The phased protocol emits a 4-key JSON; parser returns ``research_summary``."""
    body = (
        '```json\n{"code": "def solve():\\n    return 1\\n", '
        '"diff_summary": "ok", "intended_effect": "return 1", '
        '"research_summary": "Adopted divide-and-conquer from Strassen."}\n```'
    )
    code, diff_summary, intended_effect, research_summary = _parse_agent_output(body)
    assert code.startswith("def solve()")
    assert diff_summary == "ok"
    assert intended_effect == "return 1"
    assert research_summary == "Adopted divide-and-conquer from Strassen."


def test_parse_agent_output_tolerates_legacy_3_key_payload() -> None:
    """Legacy 3-key payloads (no ``research_summary``) still parse.

    The design doc preserves backward compatibility: older fixtures /
    no-tools responses that emit only ``code`` / ``diff_summary`` /
    ``intended_effect`` must keep parsing, with ``research_summary``
    defaulting to ``""``.
    """
    body = (
        '```json\n{"code": "def solve():\\n    return 0\\n", '
        '"diff_summary": "legacy", "intended_effect": "baseline"}\n```'
    )
    code, diff_summary, intended_effect, research_summary = _parse_agent_output(body)
    assert code.startswith("def solve()")
    assert diff_summary == "legacy"
    assert intended_effect == "baseline"
    assert research_summary == ""


def test_parse_agent_output_rejects_non_string_research_summary() -> None:
    """``research_summary`` must be a string when present (else loud failure)."""
    body = (
        '```json\n{"code": "def solve():\\n    return 0\\n", '
        '"diff_summary": "ok", "intended_effect": "ok", '
        '"research_summary": 42}\n```'
    )
    with pytest.raises(ClaudeAgentClientError, match="research_summary"):
        _parse_agent_output(body)


def test_sdk_client_surfaces_research_summary_on_response(
    tmp_path: Any,  # noqa: ARG001 — unused; keep tmp_path import ergonomics consistent
) -> None:
    """End-to-end: 4-key JSON flows through to ``response.research_summary``."""
    import claude_agent_sdk as sdk  # noqa: PLC0415

    four_key_body = (
        '```json\n{"code": "def solve():\\n    return 1\\n", '
        '"diff_summary": "ok", "intended_effect": "return 1", '
        '"research_summary": "Retrieved Strassen; adopted 2-level split."}\n```'
    )
    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text=four_key_body)],
        model="claude-sonnet-4-6",
    )
    result = sdk.ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="sess-research-summary",
    )

    response = _run_with_scripted_messages([assistant, result])
    assert response.research_summary == "Retrieved Strassen; adopted 2-level split."


# ---------------------------------------------------------------------------
# 9. Phased-protocol prompt rendering (research-enabled mode)
# ---------------------------------------------------------------------------


def test_render_prompt_research_mode_includes_phased_protocol() -> None:
    """``research`` mode renders the decide/gather/synthesize/mutate phases.

    Pins the exact phase headings the design doc requires so a future
    edit that accidentally drops one fails this test.
    """
    bundle = _make_bundle()
    prompt = _render_prompt(bundle, mutator_tools="research")
    # Phase scaffolding.
    assert "Research-enabled protocol (phased)" in prompt
    assert "1. Decide" in prompt
    assert "2. Gather" in prompt
    assert "3. Synthesize" in prompt
    assert "4. Mutate" in prompt
    # Bounded/purposeful guidance.
    assert "bounded" in prompt
    # Anti-cosmetic rule.
    assert "Anti-cosmetic rule" in prompt
    # Integration rule mentions research_summary.
    assert "Integration rule" in prompt
    assert "research_summary" in prompt
    # Output schema still present.
    assert "research_summary" in prompt


def test_render_prompt_no_tools_mode_omits_phased_protocol() -> None:
    """``none`` mode must NOT ship the phased protocol block.

    Prevents drift where the no-tools prompt accidentally adopts
    research-only phrasing.
    """
    bundle = _make_bundle()
    prompt = _render_prompt(bundle, mutator_tools="none")
    assert "Research-enabled protocol" not in prompt
    assert "Anti-cosmetic rule" not in prompt
    # But the 4-key schema is still advertised so no-tools responses
    # populate ``research_summary=""`` consistently.
    assert "research_summary" in prompt
