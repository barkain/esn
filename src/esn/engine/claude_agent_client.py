# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""SDK-facing boundary layer for the claude-agent-mutator backend (PR A, phase 1.1).

This module defines:

* ``MutatorInputBundle`` — the frozen, extra-forbidden Pydantic bundle that is the
  ONLY payload crossing the isolation boundary into the agent. Nothing outside
  this whitelist may reach the SDK.
* ``ClaudeAgentResponse`` — the structured return shape from an agent run.
* ``ClaudeAgentClient`` — the sync Protocol the mutator calls.
* ``ClaudeAgentSDKClient`` — the concrete ``claude_agent_sdk`` adapter. It
  constructs an isolated SDK session per call, observing the three-layer
  isolation contract (bundle-only input, fresh empty cwd, zero tools).

The SDK is imported LAZILY inside ``run_mutation`` so importing this module
does not require ``claude_agent_sdk`` to be available at construction time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from esn.engine.mutator_transcript import (
    ProposerTurnEvent,
    write_proposer_transcript,
)

_logger = logging.getLogger(__name__)


async def _stream_single_user_message(
    text: str,
) -> AsyncIterator[dict[str, Any]]:
    """Wrap a single user prompt string as an async iterable for SDK streaming mode.

    The Claude Agent SDK's ``query()`` refuses to run a ``can_use_tool``
    callback against a plain string prompt (it raises
    ``ValueError("can_use_tool callback requires streaming mode.")``). To
    keep the deny-all tool callback (Layer 3 isolation) active we must
    pass the prompt as an ``AsyncIterable[dict]`` instead.

    The dict shape is the SDK's internal user-message envelope — the same
    shape ``InternalClient.process_query`` constructs itself for string
    prompts (see claude_agent_sdk/_internal/client.py line ~145, v0.1.63).
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
    }


def _build_sdk_child_env() -> dict[str, str]:
    """Sanitize env for the Claude Agent SDK's spawned child CLI.

    - Copy current os.environ
    - Remove CLAUDECODE and CLAUDE_CODE_ENTRYPOINT so the SDK's child CLI
      does not refuse to start when this process is running inside a
      Claude Code session.
    - Set CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK=1.
    - Leave ANTHROPIC_API_KEY untouched. Subscription auth falls through
      to ~/.claude/.credentials.json when the API key is absent.
    """
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"
    return env


@dataclass(frozen=True)
class _RunResult:
    """Internal scratch tuple: the subset of stream state ``_run`` escapes with.

    Kept module-private; ``run_mutation`` immediately projects this onto the
    public ``ClaudeAgentResponse``.

    ``turn_events`` carries the ordered ``ProposerTurnEvent`` stream captured
    during ``_run()``. It is consumed by the transcript writer; the public
    ``ClaudeAgentResponse`` intentionally does NOT surface it, to keep the
    parsed-program contract byte-identical.
    """

    final_text: str
    num_turns: int
    first_line: str
    input_tokens: int | None
    output_tokens: int | None
    total_cost_usd: float | None
    turn_events: tuple[ProposerTurnEvent, ...] = field(default_factory=tuple)


@dataclass
class ProposerTranscriptContext:
    """Per-call context for writing a proposer transcript.

    ``variant_id`` defaults to ``None``; when unset, ``ClaudeAgentSDKClient``
    uses the current thread id, matching the existing
    ``RecordingLLMClient.gen_NN_tTID_{label}.md`` convention in the benchmark
    runners.
    """

    run_dir: Path
    generation: int
    variant_id: str | int | None = None


class ClaudeAgentClientError(Exception):
    """Raised when the Claude agent SDK session fails or returns malformed output.

    When the SDK's ``_run()`` completed (turns + tokens were captured) but the
    assistant text failed to parse afterwards, the error carries a
    ``partial_response`` — a ``ClaudeAgentResponse`` whose content fields are
    empty but whose accounting surface (model, turn count, tokens, cost, raw
    text) reflects what the SDK actually spent. This lets the mutator record
    real cost for failed-parse attempts so single_shot vs claude_agent_sdk
    comparisons are not biased downward on bad attempts.

    Mid-stream failures (SDK/network errors raised from inside ``_run()``)
    leave ``partial_response=None`` — accounting is genuinely absent in that
    case, and we never invent it.

    Per-call timeouts (``call_timeout_seconds`` exceeded) raise this with a
    "SDK call timed out after {N}s" message and a ``partial_response``
    carrying whatever turn/text state was accumulated before the timeout
    fired (best-effort; ``ResultMessage`` was never seen so accounting is
    typically all-None on the timeout path).
    """

    def __init__(
        self,
        *args: Any,
        partial_response: ClaudeAgentResponse | None = None,
    ) -> None:
        super().__init__(*args)
        self.partial_response = partial_response


# ---------------------------------------------------------------------------
# Isolation-boundary bundle + response
# ---------------------------------------------------------------------------


class MutatorInputBundle(BaseModel):
    """Whitelist of fields that may cross the isolation boundary.

    Frozen and ``extra="forbid"`` so accidental additions are caught at
    construction. Anything the mutator wants to send the agent MUST be a field
    here — no side-channels, no ProgramObject methods, no file paths.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain_name: str
    domain_description: str
    hard_constraints: list[str]
    allowed_imports: list[str]
    max_code_lines: int | None
    program_interface: str
    parent_code: str
    style: str
    intended_effect: str
    targeted_hypothesis_ids: list[str]
    top_hypotheses_summary: list[dict[str, str]]
    mutation_style: str
    search_mode: str
    # Advisory domain prompt-steering; ``None`` renders the fallback
    # ``(no domain-specific preference)`` in both prompt surfaces. Mirrors
    # ``DomainSpec.preferred_solution_shape``. No validator / evaluator
    # effect — pure prompt guidance.
    preferred_solution_shape: str | None = None
    # ESN spectral-analysis guidance channel, mirroring the ``LLMMutator``
    # rendering at ``src/esn/engine/mutator.py:479-480``. ``None`` renders the
    # fallback ``(no spectral guidance provided)`` in the agentic prompt.
    # Pure prompt steering — no validator / evaluator effect.
    spectral_guidance: str | None = None


class ClaudeAgentResponse(BaseModel):
    """Structured agent response returned to the mutator.

    In addition to the parsed agent output (``code``, ``diff_summary``,
    ``intended_effect``, ``agent_summary``, ``research_summary``) and
    ``agent_turn_count``, this response also surfaces accounting fields
    provided by the Claude Agent SDK's ``ResultMessage``: ``model``
    identity, token counts (when the SDK exposes a ``usage`` dict),
    ``total_cost_usd``, and the raw concatenated assistant text. Any
    field the SDK does not expose for a given run is left as ``None`` —
    we never invent data.

    ``research_summary`` is the phased-protocol synthesis channel: an empty
    string when the agent did not consult research tools, otherwise a
    short (<= 300 chars) note describing what was retrieved and what
    was adopted or rejected.
    """

    code: str
    diff_summary: str
    intended_effect: str
    agent_turn_count: int
    agent_summary: str
    research_summary: str = ""
    # Accounting surface (priority-ordered: turns -> tokens -> raw -> model).
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost_usd: float | None = None
    raw_response_text: str | None = None


# ---------------------------------------------------------------------------
# Protocol + concrete SDK adapter
# ---------------------------------------------------------------------------


@runtime_checkable
class ClaudeAgentClient(Protocol):
    """Sync facade the mutator calls. Implementations own their own async loop."""

    def run_mutation(self, bundle: MutatorInputBundle) -> ClaudeAgentResponse: ...


class ClaudeAgentSDKClient:
    """Concrete ``claude_agent_sdk`` adapter.

    Isolation contract (three layers):

    * **Layer 1 — bundle-only input.** Only the rendered prompt string (built
      from ``MutatorInputBundle``) reaches the SDK. No file attachments,
      no resource refs.
    * **Layer 2 — no workspace/project.** ``cwd`` is a fresh, empty tempdir
      allocated per call and deleted in ``finally``. No ``add_dirs``, no
      project path, ``setting_sources=None`` so user/project/local settings
      are NOT inherited.
    * **Layer 3 — no tools (default) OR research-only tools.**
      ``mutator_tools`` resolves to exactly one of two tool-list shapes;
      nothing else may leak in:

      - ``"none"`` (default): ``allowed_tools=[]``.
      - ``"research"``: ``allowed_tools=["WebSearch", "WebFetch"]`` today.
        ``"research"`` is a backend-agnostic policy — future environments
        may map it onto a different retrieval toolset (agentlib, books,
        papers) without a CLI change.

      ``mcp_servers={}``, ``skills=None``. We additionally pass a deny-all
      ``can_use_tool`` callback for ``"none"``; ``"research"`` allows only
      tools inside the configured research allowlist through the callback
      and denies anything else.
    """

    # Research-mode allowlist. The ONLY tools permitted when
    # ``mutator_tools == "research"`` today. Single source of truth:
    # runners + the mutator pass a mode STRING and the resolution happens
    # here. Extending the research toolset (e.g. agentlib_* retrieval
    # tools) means editing this tuple — no CLI change.
    _RESEARCH_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch")

    @classmethod
    def _resolve_tools(cls, mutator_tools: str) -> list[str]:
        """Map the public mode string onto the SDK ``allowed_tools`` list.

        This is the ONE place the string -> list mapping lives. Runners
        never construct the list themselves. Any value other than the two
        documented modes raises immediately so drift is loud.

        The explicit no-eval-boundary assertion at the bottom of this
        method is belt-and-suspenders: it makes tool-list drift (e.g. a
        future edit to ``_RESEARCH_TOOLS`` that accidentally adds
        ``Bash``) fail fast at construction time rather than at runtime.
        """
        if mutator_tools == "none":
            tools: list[str] = []
        elif mutator_tools == "research":
            tools = list(cls._RESEARCH_TOOLS)
        else:
            raise ValueError(f"mutator_tools must be 'none' or 'research', got {mutator_tools!r}")
        # No-eval boundary: whatever the mode resolves to, the tool list
        # MUST be a subset of the research-tool allowlist. This guards
        # every future edit to ``_RESEARCH_TOOLS`` or the resolver above.
        # Raises a ``RuntimeError`` rather than using ``assert`` so the
        # check survives ``python -O``.
        if not set(tools).issubset(set(cls._RESEARCH_TOOLS)):
            raise RuntimeError(
                f"ClaudeAgentSDKClient tool list {tools!r} escaped the "
                f"no-eval research allowlist {cls._RESEARCH_TOOLS!r}"
            )
        return tools

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_turns: int = 5,
        mutator_tools: str = "none",
        call_timeout_seconds: float = 300.0,
    ) -> None:
        # Public so the harness / mutator can surface the configured model id
        # without reaching into private attrs. ``_model`` was only referenced
        # inside this file (verified via grep), so the rename is safe.
        self.model = model
        self._max_turns = max_turns
        # Resolve at construction so bad mode strings fail BEFORE any
        # network/SDK work starts. Runners pass the mode string through;
        # the list lives only here.
        self.mutator_tools = mutator_tools
        self._allowed_tools: list[str] = self._resolve_tools(mutator_tools)
        # Per-call wall-clock timeout for the FULL SDK stream consumption.
        # Default 300s. A standalone repro demonstrated sonnet-4-6
        # wedging for 300+s without producing a
        # ``ResultMessage`` — without this trap, the mutator hangs the
        # entire harness instead of recording a failed candidate. The
        # timeout wraps the WHOLE async-for over ``query(...)``, so a hung
        # generator (no messages flowing) is interrupted at the bound.
        self.call_timeout_seconds = float(call_timeout_seconds)
        # Optional transcript context. Benchmark runners call
        # ``set_transcript_context`` each generation so a per-attempt
        # proposer transcript is dropped under ``run_dir/transcripts``.
        # When unset, no transcript is written — this preserves the
        # contract for unit tests and for consumers that don't want the
        # side effect.
        self._transcript_ctx: ProposerTranscriptContext | None = None

    # ------------------------------------------------------------------
    # Transcript context (side-effect plumbing; no behavior impact)
    # ------------------------------------------------------------------

    def set_transcript_context(
        self,
        *,
        run_dir: Path | str,
        generation: int,
        variant_id: str | int | None = None,
    ) -> None:
        """Enable per-attempt proposer transcripts for subsequent ``run_mutation`` calls.

        Once set, every ``run_mutation`` writes a ``gen_NN_tXXX_proposer.md``
        file under ``run_dir/transcripts``. The benchmark runner is expected
        to bump ``generation`` between generations; ``variant_id`` defaults
        to the calling thread id (matching the predictor/analyzer
        ``RecordingLLMClient`` convention).
        """
        self._transcript_ctx = ProposerTranscriptContext(
            run_dir=Path(run_dir),
            generation=int(generation),
            variant_id=variant_id,
        )

    def clear_transcript_context(self) -> None:
        """Disable proposer-transcript writing for subsequent calls."""
        self._transcript_ctx = None

    def _write_proposer_transcript_if_configured(self, run_result: _RunResult) -> None:
        """Write ``gen_NN_tXXX_proposer.md`` for this attempt, if configured.

        Side-effecting helper; never raises — a transcript-write failure
        must not break the mutation path. The context must be set via
        ``set_transcript_context`` first; when unset, this is a no-op.
        """
        ctx = self._transcript_ctx
        if ctx is None:
            return
        variant_id = ctx.variant_id if ctx.variant_id is not None else threading.get_ident()
        # Best-effort: pull ``research_summary`` out of the final text so
        # the transcript's dedicated section reflects what the agent
        # actually returned. Parse failures here are silent — the
        # transcript must still be written for debugging, with
        # ``research_summary=""`` in that case.
        research_summary_for_transcript = ""
        try:
            (_c, _d, _i, research_summary_for_transcript) = _parse_agent_output(
                run_result.final_text
            )
        except ClaudeAgentClientError:
            research_summary_for_transcript = ""
        try:
            write_proposer_transcript(
                run_dir=ctx.run_dir,
                generation=ctx.generation,
                variant_id=variant_id,
                turn_events=list(run_result.turn_events),
                final_response_text=run_result.final_text,
                mutator_tools=self.mutator_tools,
                backend="claude_agent_sdk",
                model=self.model,
                research_summary=research_summary_for_transcript,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "proposer transcript write failed (gen=%s, variant=%s): %s",
                ctx.generation,
                variant_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_mutation(self, bundle: MutatorInputBundle) -> ClaudeAgentResponse:
        """Run one mutation round-trip through the agent SDK and return the parsed result."""
        # Lazy imports: keep top-level module import cheap and SDK-optional.
        from claude_agent_sdk import (  # noqa: PLC0415
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKError,
            PermissionResultAllow,
            PermissionResultDeny,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        prompt = _render_prompt(bundle, mutator_tools=self.mutator_tools)
        tempdir = tempfile.mkdtemp(prefix="esn_claude_agent_")
        try:
            allowed_tools = list(self._allowed_tools)
            allowed_set = set(allowed_tools)
            research_allowlist = set(self._RESEARCH_TOOLS)

            async def _gate_tools(
                tool_name: str,
                _input: dict[str, Any],
                _context: Any,
            ) -> PermissionResultDeny | Any:
                """Gate every tool call against the resolved allowlist.

                - ``"none"`` mode: ``allowed_set`` is empty -> everything
                  is denied.
                - ``"research"`` mode: only tools in the research
                  allowlist (today: ``WebSearch`` / ``WebFetch``) are
                  permitted; anything else (including evaluator tools
                  like Bash / Read / Write / Edit) is denied.
                """
                if tool_name in allowed_set and tool_name in research_allowlist:
                    return PermissionResultAllow()
                return PermissionResultDeny(
                    message=(
                        f"tool {tool_name!r} denied by mutator_tools="
                        f"{self.mutator_tools!r} isolation contract"
                    ),
                    interrupt=True,
                )

            options = ClaudeAgentOptions(
                model=self.model,
                max_turns=self._max_turns,
                # Layer 3: tool-policy resolution happened in __init__.
                # ``"none"`` -> []; ``"research"`` -> today
                # ``["WebSearch", "WebFetch"]`` (future retrieval tools
                # plug into the same research mode without a CLI change).
                allowed_tools=allowed_tools,
                disallowed_tools=[],  # everything outside allowed_tools is denied by the gate
                mcp_servers={},  # no external MCP servers
                skills=None,  # no skills loaded
                cwd=tempdir,  # Layer 2: fresh empty workspace
                add_dirs=[],  # no extra directories exposed
                setting_sources=None,  # do NOT inherit user/project/local settings
                can_use_tool=_gate_tools,
                env=_build_sdk_child_env(),  # sanitize Claude Code session env
            )

            # Mutable accumulator: written incrementally by ``_run_inner`` so
            # that on ``asyncio.TimeoutError`` we can still build a partial
            # ``_RunResult`` from whatever was collected before the wedge.
            # The inner coroutine's return value is the canonical result on
            # the success path; on timeout we fall back to this scratch.
            partial_state: dict[str, Any] = {
                "final_text": "",
                "num_turns": 0,
                "first_line": "",
                "input_tokens": None,
                "output_tokens": None,
                "total_cost_usd": None,
                "turn_events": [],
            }

            async def _run_inner() -> _RunResult:
                # Aliases into the shared accumulator so partial transcript
                # state survives a timeout. We deliberately read/write
                # through the dict rather than rebinding locals, so the
                # outer ``partial_state`` stays in sync.
                turn_events: list[ProposerTurnEvent] = partial_state["turn_events"]
                # The prompt MUST be passed as an AsyncIterable (streaming
                # mode) because ``options.can_use_tool`` is set — the SDK
                # refuses the string-prompt + callback combination. See
                # ``_stream_single_user_message`` for the wrapper rationale
                # and the SDK cross-reference.
                prompt_stream = _stream_single_user_message(prompt)
                async for message in query(prompt=prompt_stream, options=options):
                    if isinstance(message, AssistantMessage):
                        # Concatenate all text blocks in this assistant turn.
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                cur = partial_state["final_text"]
                                partial_state["final_text"] = (
                                    block.text if not cur else (cur + "\n" + block.text)
                                )
                                if not partial_state["first_line"] and block.text.strip():
                                    partial_state["first_line"] = block.text.strip().splitlines()[0]
                                turn_events.append(
                                    ProposerTurnEvent(
                                        kind="text",
                                        text=block.text,
                                    )
                                )
                            elif isinstance(block, ToolUseBlock):
                                turn_events.append(
                                    ProposerTurnEvent(
                                        kind="tool_use",
                                        tool_name=block.name,
                                        tool_use_id=block.id,
                                        tool_input=dict(block.input)
                                        if isinstance(block.input, dict)
                                        else {"_raw": str(block.input)},
                                    )
                                )
                    elif isinstance(message, UserMessage):
                        # The SDK surfaces tool_result payloads as blocks
                        # inside a synthetic UserMessage after the tool
                        # executes. ``content`` may be a string (legacy
                        # shape) or a list of blocks.
                        if isinstance(message.content, list):
                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    # Resolve the tool name via the matching
                                    # tool_use_id (captured earlier in the
                                    # same stream). Writer also does this
                                    # fallback, so either is safe.
                                    name_for_id = ""
                                    for prev in turn_events:
                                        if (
                                            prev.kind == "tool_use"
                                            and prev.tool_use_id == block.tool_use_id
                                        ):
                                            name_for_id = prev.tool_name
                                            break
                                    turn_events.append(
                                        ProposerTurnEvent(
                                            kind="tool_result",
                                            tool_name=name_for_id,
                                            tool_use_id=block.tool_use_id,
                                            tool_result_content=block.content,
                                            tool_result_is_error=block.is_error,
                                        )
                                    )
                    elif isinstance(message, ResultMessage):
                        partial_state["num_turns"] = message.num_turns
                        # Accounting fields: pass through the SDK's values
                        # verbatim. ``usage`` may be ``None`` or missing keys
                        # depending on model/stream config, so use ``.get``.
                        usage = getattr(message, "usage", None)
                        if isinstance(usage, dict):
                            # Prefer Anthropic-style keys, fall back to the
                            # OpenAI-style aliases seen in some SDK paths.
                            input_tokens = usage.get("input_tokens")
                            if input_tokens is None:
                                input_tokens = usage.get("prompt_tokens")
                            output_tokens = usage.get("output_tokens")
                            if output_tokens is None:
                                output_tokens = usage.get("completion_tokens")
                            partial_state["input_tokens"] = input_tokens
                            partial_state["output_tokens"] = output_tokens
                        partial_state["total_cost_usd"] = getattr(message, "total_cost_usd", None)
                # Fallback if ResultMessage never arrived.
                return _RunResult(
                    final_text=partial_state["final_text"],
                    num_turns=partial_state["num_turns"],
                    first_line=partial_state["first_line"],
                    input_tokens=partial_state["input_tokens"],
                    output_tokens=partial_state["output_tokens"],
                    total_cost_usd=partial_state["total_cost_usd"],
                    turn_events=tuple(turn_events),
                )

            async def _run_with_timeout() -> _RunResult:
                # ``asyncio.wait_for`` wraps the FULL stream consumption.
                # When the SDK child wedges (no messages flowing), this
                # raises ``asyncio.TimeoutError`` at the deadline rather
                # than letting the harness hang indefinitely. A standalone
                # repro demonstrated the failure mode this guards against.
                return await asyncio.wait_for(
                    _run_inner(),
                    timeout=self.call_timeout_seconds,
                )

            try:
                run_result = asyncio.run(_run_with_timeout())
            except asyncio.TimeoutError as exc:
                # Build a best-effort partial response from whatever the
                # accumulator captured before the wedge. ``num_turns`` /
                # tokens / cost are typically still ``None`` here because
                # ``ResultMessage`` was never emitted; that is honest —
                # we never invent accounting we did not see.
                partial = ClaudeAgentResponse(
                    code="",
                    diff_summary="",
                    intended_effect="",
                    research_summary="",
                    agent_turn_count=int(partial_state["num_turns"]),
                    agent_summary=(
                        partial_state["first_line"]
                        or f"timed out after {self.call_timeout_seconds}s"
                    ),
                    model=self.model,
                    input_tokens=partial_state["input_tokens"],
                    output_tokens=partial_state["output_tokens"],
                    total_cost_usd=partial_state["total_cost_usd"],
                    raw_response_text=partial_state["final_text"] or None,
                )
                # Still write the proposer transcript on timeout so a
                # debugger can see what (if anything) the agent produced
                # before the wedge. Best-effort; logged-and-swallowed —
                # transcript I/O errors must not mask the timeout.
                try:
                    self._write_proposer_transcript_if_configured(
                        _RunResult(
                            final_text=partial_state["final_text"],
                            num_turns=int(partial_state["num_turns"]),
                            first_line=partial_state["first_line"],
                            input_tokens=partial_state["input_tokens"],
                            output_tokens=partial_state["output_tokens"],
                            total_cost_usd=partial_state["total_cost_usd"],
                            turn_events=tuple(partial_state["turn_events"]),
                        )
                    )
                except Exception as transcript_exc:  # noqa: BLE001
                    _logger.warning(
                        "proposer transcript write on timeout failed: %s",
                        transcript_exc,
                    )
                raise ClaudeAgentClientError(
                    f"SDK call timed out after {self.call_timeout_seconds}s",
                    partial_response=partial,
                ) from exc
            except ClaudeSDKError as exc:
                raise ClaudeAgentClientError(f"SDK error: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise ClaudeAgentClientError(f"agent run failed: {exc}") from exc

            # Write the proposer transcript for this attempt (if context
            # is configured). Writes happen regardless of parse success —
            # a failed-parse attempt is the MOST useful one to inspect.
            # Failures here are logged and swallowed; a transcript-write
            # I/O error must not break the mutation path.
            self._write_proposer_transcript_if_configured(run_result)

            # Parse-after-run: if parsing fails here, ``_run()`` already spent
            # the SDK turns/tokens. Attach a partial ``ClaudeAgentResponse``
            # carrying the accounting so the mutator can record real cost on
            # failed-parse attempts. Content fields are empty strings (not
            # invented data).
            try:
                (
                    code,
                    diff_summary,
                    intended_effect,
                    research_summary,
                ) = _parse_agent_output(run_result.final_text)
            except ClaudeAgentClientError as parse_exc:
                partial = ClaudeAgentResponse(
                    code="",
                    diff_summary="",
                    intended_effect="",
                    research_summary="",
                    agent_turn_count=run_result.num_turns,
                    agent_summary=(run_result.first_line or f"turns={run_result.num_turns}"),
                    model=self.model,
                    input_tokens=run_result.input_tokens,
                    output_tokens=run_result.output_tokens,
                    total_cost_usd=run_result.total_cost_usd,
                    raw_response_text=run_result.final_text or None,
                )
                raise ClaudeAgentClientError(
                    str(parse_exc),
                    partial_response=partial,
                ) from parse_exc

            summary = run_result.first_line or f"turns={run_result.num_turns}"
            return ClaudeAgentResponse(
                code=code,
                diff_summary=diff_summary,
                intended_effect=intended_effect,
                research_summary=research_summary,
                agent_turn_count=run_result.num_turns,
                agent_summary=summary,
                model=self.model,
                input_tokens=run_result.input_tokens,
                output_tokens=run_result.output_tokens,
                total_cost_usd=run_result.total_cost_usd,
                raw_response_text=run_result.final_text or None,
            )
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompt rendering + response parsing (module-private helpers)
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _runtime_budget_hint_for_bundle(bundle: MutatorInputBundle) -> str:
    """Render the wall-clock runtime-budget hint, domain-aware.

    The bundle's ``allowed_imports`` is a list of strings. When it is
    empty OR contains ``"time"``, the hint concretely points at
    ``time.time()``. An empty list here signals the mutator passed
    "unrestricted" (``None`` on the DomainSpec collapses to ``[]`` on the
    bundle) — consistent with how the rendered ``allowed_imports`` block
    above treats empty as unrestricted. Otherwise, render a generic
    iteration-cap variant that names NO specific stdlib module, so
    AST-restricted domains do not reject code induced by the hint.
    """
    imports = bundle.allowed_imports
    allows_time = not imports or "time" in imports
    if allows_time:
        return (
            "Your candidate runs under a wall-clock evaluation budget enforced by the harness. "
            "When your proposed code includes search, optimization, annealing, or any other loop "
            "whose iteration count could grow large, add an explicit elapsed-time guard using "
            "`time.time()` (e.g., `start = time.time(); while time.time() - start < BUDGET: ...`). "
            "Leave a safety margin below the benchmark's cap rather than running right up to it. "
            "Prefer finishing with a partial-but-scored result over being killed mid-search.\n"
        )
    return (
        "Your candidate runs under a wall-clock evaluation budget enforced by the harness. "
        "When your proposed code includes search, optimization, annealing, or any loop whose "
        "iteration count could grow large, give it an explicit iteration cap or early-exit "
        "condition. Leave a safety margin below the benchmark's cap rather than running right "
        "up to it. Prefer finishing with a partial-but-scored result over being killed "
        "mid-search.\n"
    )


def _render_prompt(bundle: MutatorInputBundle, mutator_tools: str = "none") -> str:
    """Render the task prompt sent to the agent.

    The prompt carries the isolation rules up-front (tool policy, no eval,
    json-only output), then the domain/mutation context, then the parent
    code, then the final return-shape instruction.

    ``mutator_tools`` selects which tool-use wording the prompt ships with:

    * ``"none"`` — explicitly declares no tools, asks for a direct final
      answer in the 4-key JSON schema (``research_summary`` is ``""``).
    * ``"research"`` — the phased research-enabled protocol:
      decide -> gather -> synthesize -> mutate. The agent MAY invoke the
      research tools that the SDK exposes for this environment (today:
      ``WebSearch`` / ``WebFetch``).

    Anything else falls back to the no-tools phrasing.
    """
    constraints_block = (
        "\n".join(f"- {c}" for c in bundle.hard_constraints)
        if bundle.hard_constraints
        else "(none)"
    )
    imports_block = (
        ", ".join(sorted(bundle.allowed_imports))
        if bundle.allowed_imports
        else "(any — unrestricted)"
    )
    max_lines_block = (
        str(bundle.max_code_lines) if bundle.max_code_lines is not None else "(no limit)"
    )
    targeted_ids_block = (
        ", ".join(bundle.targeted_hypothesis_ids) if bundle.targeted_hypothesis_ids else "(none)"
    )

    hypotheses_lines: list[str] = []
    for hyp in bundle.top_hypotheses_summary:
        hyp_id = hyp.get("id", "?")
        hyp_text = hyp.get("text", "")
        hypotheses_lines.append(f"- [{hyp_id}] {hyp_text}")
    hypotheses_block = "\n".join(hypotheses_lines) if hypotheses_lines else "(none)"

    # Tool-use + protocol wording. The ``"research"`` branch stages the
    # work explicitly (decide / gather / synthesize / mutate) so research
    # is a first-class phase rather than an optional side path that
    # competes with immediate JSON completion.
    if mutator_tools == "research":
        protocol_block = (
            "# Research-enabled protocol (phased)\n"
            "You are running in research-enabled mode. Follow these phases, in order:\n"
            "1. Decide — first decide whether external research would materially "
            "help this specific mutation. If it would not, skip phases 2 and 3.\n"
            "2. Gather — if research is needed, use the available research tools "
            "in a bounded, purposeful way. Gather a small amount of directly "
            "relevant evidence; do not run a literature review. Do not spend "
            "the whole mutation budget on research.\n"
            "3. Synthesize — reduce what you gathered into a short actionable "
            "plan. Record it in the `research_summary` output field.\n"
            "4. Mutate — produce the final candidate program using that plan.\n"
            "\n"
            "Anti-cosmetic rule: do NOT use research only to attach citations or "
            "a sources footer. Retrieved information should change the algorithm, "
            "implementation, or constraints handling — not decorate the answer.\n"
            "Integration rule: if research influenced the candidate, reflect that "
            "in `research_summary` (what was found, what was adopted, what was "
            "rejected or ignored). An optional brief code comment near the "
            "changed logic is welcome when useful.\n"
            "\n"
            "# Isolation rules (binding)\n"
            "- You may use ONLY the research tools that this environment exposes "
            "(today: `WebSearch` and `WebFetch`). Do not attempt any other tools.\n"
            "- No Bash, no file I/O, no workspace traversal, no evaluator access.\n"
        )
    else:
        protocol_block = (
            "# Isolation rules (binding)\n"
            "- You have NO tools available. Do not attempt tool calls.\n"
        )

    return (
        f"{protocol_block}"
        "- You MUST NOT request to run, compile, or evaluate the program.\n"
        "- You MUST NOT read or write files; no filesystem exists for you.\n"
        "- Your entire final answer MUST be a single fenced ```json block with\n"
        "  exactly these four keys: code, diff_summary, intended_effect, "
        "research_summary.\n"
        "- code: the COMPLETE replacement Python program (string).\n"
        "- diff_summary: <= 200 chars describing what changed from the parent.\n"
        "- intended_effect: <= 200 chars describing the behavioural goal.\n"
        '- research_summary: <= 300 chars. Empty string ("") when no research '
        "was used. Otherwise state what was found, what idea was adopted, and "
        "what was rejected or ignored if relevant.\n"
        "- Return ONLY the fenced json block as your final message, nothing else.\n"
        "\n"
        "# Domain\n"
        f"name: {bundle.domain_name}\n"
        f"description: {bundle.domain_description}\n"
        f"program_interface: {bundle.program_interface}\n"
        f"max_code_lines: {max_lines_block}\n"
        f"allowed_imports: {imports_block}\n"
        "hard_constraints:\n"
        f"{constraints_block}\n"
        "\n"
        "# Preferred solution shape\n"
        f"{bundle.preferred_solution_shape or '(no domain-specific preference)'}\n"
        "\n"
        "# Spectral guidance\n"
        f"{bundle.spectral_guidance or '(no spectral guidance provided)'}\n"
        "\n"
        "# Runtime budget\n"
        f"{_runtime_budget_hint_for_bundle(bundle)}"
        "\n"
        "# Mutation task\n"
        f"style: {bundle.style}\n"
        f"mutation_style: {bundle.mutation_style}\n"
        f"search_mode: {bundle.search_mode}\n"
        f"intended_effect: {bundle.intended_effect or '(improve score while staying valid)'}\n"
        f"targeted_hypothesis_ids: {targeted_ids_block}\n"
        "\n"
        "# Top hypotheses\n"
        f"{hypotheses_block}\n"
        "\n"
        "# Parent program\n"
        "```python\n"
        f"{bundle.parent_code}\n"
        "```\n"
        "\n"
        "Return ONLY the fenced json block as your final message, nothing else."
    )


def _parse_agent_output(text: str) -> tuple[str, str, str, str]:
    """Extract ``(code, diff_summary, intended_effect, research_summary)``
    from a fenced json block.

    ``research_summary`` is required in the parsed JSON object but may be
    the empty string when no research was used (or in no-tools mode).
    Legacy 3-key payloads (no ``research_summary`` key) are tolerated and
    mapped to ``research_summary=""`` so older fixtures and no-tools
    responses still parse; the prompt always asks for the 4-key schema.

    Raises ``ClaudeAgentClientError`` on any parse / missing-key /
    wrong-type error so the mutator can handle it as a failed attempt.
    """
    if not text or not text.strip():
        raise ClaudeAgentClientError("malformed agent response: empty text")

    match = _JSON_FENCE_RE.search(text)
    if match is not None:
        payload = match.group(1).strip()
    else:
        # Permit a bare JSON object too — some models drop the fence.
        stripped = text.strip()
        if stripped.startswith("{"):
            payload = stripped
        else:
            raise ClaudeAgentClientError("malformed agent response: no fenced ```json block found")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ClaudeAgentClientError(f"malformed agent response: JSON decode error: {exc}") from exc

    if not isinstance(data, dict):
        raise ClaudeAgentClientError("malformed agent response: top-level JSON is not an object")

    code = data.get("code")
    diff_summary = data.get("diff_summary", "")
    intended_effect = data.get("intended_effect", "")
    research_summary = data.get("research_summary", "")

    if not isinstance(code, str) or not code.strip():
        raise ClaudeAgentClientError(
            "malformed agent response: 'code' missing, non-string, or empty"
        )
    if not isinstance(diff_summary, str):
        raise ClaudeAgentClientError("malformed agent response: 'diff_summary' is not a string")
    if not isinstance(intended_effect, str):
        raise ClaudeAgentClientError("malformed agent response: 'intended_effect' is not a string")
    if not isinstance(research_summary, str):
        raise ClaudeAgentClientError("malformed agent response: 'research_summary' is not a string")

    return code, diff_summary, intended_effect, research_summary


# Re-export ``Field`` to silence lint if unused; keep at bottom to avoid cycles.
__all__ = [
    "ClaudeAgentClient",
    "ClaudeAgentClientError",
    "ClaudeAgentResponse",
    "ClaudeAgentSDKClient",
    "MutatorInputBundle",
    "ProposerTranscriptContext",
]

# ``Field`` imported for future extensions (e.g., validators). Mark as used.
_ = Field
