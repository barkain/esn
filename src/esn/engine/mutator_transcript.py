# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Proposer-turn transcript writer for the ClaudeAgentMutator path.

Motivation
----------

Existing predictor/analyzer transcripts (written by ``RecordingLLMClient`` in
benchmark runners) let us inspect the prompts / responses for those phases.
The Claude-agent-backed mutator path has no equivalent: when
``--mutator-tools research`` is enabled, we can confirm the tool allowlist,
but we cannot see whether the exposed research tools (today: ``WebSearch``
/ ``WebFetch``) were INVOKED, what the queries or URLs were, or what content
came back.

This module provides:

* ``ProposerTurnEvent`` — a lightweight, test-friendly record of a single
  turn event (``tool_use``, ``tool_result``, or ``text``).
* ``summarize_tool_result_content`` — bounded-size summarization so a
  transcript stays in the low-KB to few-KB range per attempt.
* ``write_proposer_transcript`` — writes
  ``{run_dir}/transcripts/gen_{NN:02d}_t{variant_id}_proposer.md`` in a shape
  that matches the existing predictor/analyzer convention.

Design notes
------------

The writer is a PURE function in ``ProposerTurnEvent`` + metadata. It does
NOT import ``claude_agent_sdk``. The client layer is responsible for
translating SDK ``ToolUseBlock`` / ``ToolResultBlock`` / ``TextBlock`` objects
into ``ProposerTurnEvent`` records before calling the writer. This keeps the
transcript writer unit-testable without the SDK installed, and keeps SDK
types from leaking past the client boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from esn.engine.mutator_framing import mutator_mode_label


# Size budgets for tool-result summarization.
_WEBSEARCH_MAX_RESULTS = 10
_WEBSEARCH_SNIPPET_CHARS = 200
_WEBSEARCH_TOTAL_BUDGET_BYTES = 2048
_WEBFETCH_TEXT_CHARS = 2000
_GENERIC_RESULT_BYTES = 1024


EventKind = Literal["tool_use", "tool_result", "text"]


@dataclass(frozen=True)
class ProposerTurnEvent:
    """One event captured from the mutator's agent stream.

    Exactly one of the following field groups is populated per kind:

    * ``kind == "tool_use"``: ``tool_name``, ``tool_use_id``, ``tool_input``
      (arbitrary dict — the SDK's ``ToolUseBlock.input``).
    * ``kind == "tool_result"``: ``tool_use_id``, ``tool_result_content``
      (string | list[dict] | None), ``tool_result_is_error``. The ``tool_name``
      is carried through from the matching ``tool_use`` so the writer can pick
      the right summarization rule without a second lookup.
    * ``kind == "text"``: ``text`` (assistant-visible text).
    """

    kind: EventKind
    tool_name: str = ""
    tool_use_id: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_result_content: Any = None
    tool_result_is_error: bool | None = None
    text: str = ""


def summarize_tool_result_content(tool_name: str, content: Any) -> str:
    """Return a bounded-size, readable summary of a tool_result payload.

    Rules (from the PR brief):

    * ``WebSearch`` — if the raw payload serializes under
      ~2 KB, keep it verbatim. Otherwise keep the first 10 results,
      each with title + URL + first 200 chars of snippet.
    * ``WebFetch`` — URL + status + first 2 KB of fetched text, with an
      explicit truncation marker when truncated.
    * Anything else — full content under 1 KB; otherwise first 1 KB +
      truncation marker.

    The return is always a JSON-like fenced-markdown body the writer can
    embed directly under the relevant section.
    """
    if tool_name == "WebSearch":
        return _summarize_web_search(content)
    if tool_name == "WebFetch":
        return _summarize_web_fetch(content)
    return _summarize_generic(content)


def _json_dumps_compact(obj: Any) -> str:
    """Serialize to JSON with a deterministic width, falling back to ``str``.

    Used inside the summarizer for both size measurement and emission.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def _summarize_web_search(content: Any) -> str:
    """Summarize a ``WebSearch`` tool_result.

    The SDK delivers ``content`` as either a string (unstructured text)
    or a list of blocks (the Anthropic-style ``[{"type": "text", ...}]``).
    We don't try to fully re-parse the search engine's JSON — we just
    preserve the first ~10 result-like entries when the payload looks like
    a list of dicts, else fall through to the generic truncator.
    """
    raw = _json_dumps_compact(content)
    if len(raw.encode("utf-8")) <= _WEBSEARCH_TOTAL_BUDGET_BYTES:
        return raw

    # Large payload — try to cap at the first N entries when the shape
    # is a list. Otherwise fall back to the generic byte-truncator.
    if isinstance(content, list):
        truncated_results: list[dict[str, Any]] = []
        for item in content[:_WEBSEARCH_MAX_RESULTS]:
            if isinstance(item, dict):
                # Try the common Anthropic WebSearch keys. Missing keys
                # degrade gracefully to whatever the dict exposes.
                entry = {
                    "title": item.get("title", "")[:200]
                    if isinstance(item.get("title"), str)
                    else item.get("title"),
                    "url": item.get("url", item.get("link", "")),
                    "snippet": (
                        item.get("snippet", item.get("description", "") or "")[
                            :_WEBSEARCH_SNIPPET_CHARS
                        ]
                        if isinstance(item.get("snippet", item.get("description", "")), str)
                        else ""
                    ),
                }
                truncated_results.append(entry)
            else:
                truncated_results.append({"value": str(item)[:_WEBSEARCH_SNIPPET_CHARS]})
        summary = {
            "_truncated": True,
            "_note": (
                f"kept first {len(truncated_results)} of {len(content)} entries; "
                f"snippets capped at {_WEBSEARCH_SNIPPET_CHARS} chars"
            ),
            "results": truncated_results,
        }
        return _json_dumps_compact(summary)

    return _summarize_generic(content)


def _summarize_web_fetch(content: Any) -> str:
    """Summarize a ``WebFetch`` tool_result: URL + status + first ~2 KB text.

    The ``WebFetch`` tool can emit either structured dicts
    (``{"url", "status", "text"}``-ish) or plain strings. We preserve
    whatever keys we find, capping the text portion at ~2 KB and adding an
    explicit truncation marker when the cap fires.
    """
    if isinstance(content, dict):
        text = content.get("text", content.get("content", ""))
        if isinstance(text, str) and len(text) > _WEBFETCH_TEXT_CHARS:
            summary = {
                "url": content.get("url"),
                "status": content.get("status"),
                "text": text[:_WEBFETCH_TEXT_CHARS],
                "_truncated": True,
                "_note": (f"fetched text truncated to first {_WEBFETCH_TEXT_CHARS} chars"),
            }
            return _json_dumps_compact(summary)
        return _json_dumps_compact(content)

    if isinstance(content, str):
        if len(content) > _WEBFETCH_TEXT_CHARS:
            return (
                content[:_WEBFETCH_TEXT_CHARS]
                + f"\n\n[... truncated to first {_WEBFETCH_TEXT_CHARS} chars ...]"
            )
        return content

    # list-of-blocks or anything else — run through the generic truncator.
    return _summarize_generic(content)


def _summarize_generic(content: Any) -> str:
    """Catch-all: full content under 1 KB, else first 1 KB + marker."""
    if isinstance(content, str):
        raw = content
    else:
        raw = _json_dumps_compact(content)
    encoded = raw.encode("utf-8")
    if len(encoded) <= _GENERIC_RESULT_BYTES:
        return raw
    # Byte-bounded truncation, re-decoded with ``errors='ignore'`` so a
    # multi-byte codepoint that straddles the boundary doesn't blow up.
    truncated = encoded[:_GENERIC_RESULT_BYTES].decode("utf-8", errors="ignore")
    return truncated + f"\n\n[... truncated to first {_GENERIC_RESULT_BYTES} bytes ...]"


def _count_tool_invocations(events: list[ProposerTurnEvent]) -> dict[str, int]:
    """Count ``tool_use`` events grouped by tool name (for the header line)."""
    counts: dict[str, int] = {}
    for ev in events:
        if ev.kind == "tool_use":
            counts[ev.tool_name] = counts.get(ev.tool_name, 0) + 1
    return counts


def _format_tool_counts(counts: dict[str, int]) -> str:
    """Render ``{"WebSearch": 2, "WebFetch": 1}`` as ``"2 WebSearch, 1 WebFetch"``."""
    if not counts:
        return "none"
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))


def write_proposer_transcript(
    *,
    run_dir: Path,
    generation: int,
    variant_id: str | int,
    turn_events: list[ProposerTurnEvent],
    final_response_text: str,
    mutator_tools: str,
    backend: str,
    model: str,
    research_summary: str = "",
) -> Path:
    """Write a ``gen_{NN:02d}_t{variant_id}_proposer.md`` transcript.

    Parameters
    ----------
    run_dir
        The benchmark run directory. Transcripts go under
        ``run_dir / "transcripts"``; the subdir is created if missing.
    generation
        Generation index (0-based or 1-based — caller decides). Rendered
        with ``:02d``.
    variant_id
        Per-attempt discriminator. Typically a thread id (matching the
        existing predictor/analyzer convention), but any stringifiable
        value works.
    turn_events
        List of ``ProposerTurnEvent`` captured during the mutator's agent
        stream, in order. May be empty for the no-tools success path
        (only the final response is interesting then).
    final_response_text
        The verbatim concatenated assistant text that was parsed into a
        program. Dumped raw into a code block — the writer does NOT
        prettify or re-wrap it.
    mutator_tools
        ``"none"`` or ``"research"`` — the configured tool mode (header only).
    backend
        Mutator backend label (e.g., ``"claude_agent_sdk"``) for the header.
    model
        Mutator model id string (e.g., ``"claude-haiku-4-5-20251001"``).

    Returns the path the transcript was written to.
    """
    transcripts_dir = Path(run_dir) / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    path = transcripts_dir / f"gen_{int(generation):02d}_t{variant_id}_proposer.md"

    counts = _count_tool_invocations(turn_events)
    # Research mode: enumerate the concrete retrieval tools available in
    # this environment. Kept in lockstep with
    # ``ClaudeAgentSDKClient._RESEARCH_TOOLS`` and
    # ``mutator_framing._research_tools_available``.
    research_tools_available = "WebSearch, WebFetch" if mutator_tools == "research" else "none"

    lines: list[str] = []
    lines.append(f"# Proposer turn — gen {int(generation):02d} / variant {variant_id}")
    lines.append("")
    # Use the shared framing helper so this writer stays in lockstep with
    # the runner consoles + report.md header (canonical labels:
    # "agentic mutation (no tools)" / "agentic mutation (research-enabled)").
    mode_label = mutator_mode_label("claude_agent_sdk", tools=mutator_tools)
    lines.append(f"**Mutator mode**: {mode_label}")
    lines.append(f"**Backend**: {backend}")
    lines.append(f"**Model**: {model}")
    lines.append(f"**Mutator tools**: {mutator_tools}")
    lines.append(f"**Research tools available**: {research_tools_available}")
    lines.append(f"**Tool invocations**: {_format_tool_counts(counts)}")
    lines.append("")

    # Turn events section — only rendered when there is at least one
    # tool_use / tool_result event. Pure-text events feed into the
    # "Final response" section below; emitting them here too would
    # duplicate the content.
    has_tool_events = any(ev.kind in ("tool_use", "tool_result") for ev in turn_events)
    if has_tool_events:
        lines.append("## Turn events")
        lines.append("")
        tool_use_idx = 0
        tool_result_idx = 0
        # Map tool_use_id -> tool_name so we can attach the name to the
        # matching tool_result for summarizer dispatch.
        id_to_name: dict[str, str] = {}
        for ev in turn_events:
            if ev.kind == "tool_use":
                tool_use_idx += 1
                id_to_name[ev.tool_use_id] = ev.tool_name
                lines.append(f"### Tool use #{tool_use_idx} — {ev.tool_name}")
                lines.append("")
                # Named fields we promote to top-level markdown lines so
                # the eye can scan them without reading the JSON body.
                if ev.tool_name == "WebSearch":
                    query = ev.tool_input.get("query", "")
                    lines.append(f"**Query**: {json.dumps(query, ensure_ascii=False)}")
                elif ev.tool_name == "WebFetch":
                    url = ev.tool_input.get("url", "")
                    prompt = ev.tool_input.get("prompt", "")
                    lines.append(f"**URL**: {url}")
                    if prompt:
                        lines.append(f"**Prompt**: {json.dumps(prompt, ensure_ascii=False)}")
                lines.append("")
                lines.append("**Input**:")
                lines.append("")
                lines.append("```json")
                lines.append(_json_dumps_compact(ev.tool_input))
                lines.append("```")
                lines.append("")
            elif ev.kind == "tool_result":
                tool_result_idx += 1
                tool_name_for_summary = ev.tool_name or id_to_name.get(ev.tool_use_id, "")
                lines.append(f"### Tool result #{tool_result_idx}")
                lines.append("")
                if ev.tool_result_is_error:
                    lines.append("**is_error**: true")
                    lines.append("")
                summary = summarize_tool_result_content(
                    tool_name_for_summary, ev.tool_result_content
                )
                lines.append("```")
                lines.append(summary)
                lines.append("```")
                lines.append("")

    # Research synthesis: surface the ``research_summary`` field prominently
    # between turn events and the final response, so readers of the
    # transcript do NOT need to parse the final JSON block to see what
    # the agent concluded from any research it did. Only emitted for
    # research mode — in no-tools mode there is nothing to synthesize.
    if mutator_tools == "research":
        lines.append("## Research synthesis")
        lines.append("")
        if research_summary.strip():
            lines.append(research_summary.strip())
        else:
            lines.append("_(agent reported no research used)_")
        lines.append("")

    lines.append("## Final response (parsed into program)")
    lines.append("")
    lines.append("```")
    # Verbatim — per the brief, do NOT prettify or re-wrap.
    lines.append(final_response_text)
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines))
    return path


__all__ = [
    "ProposerTurnEvent",
    "summarize_tool_result_content",
    "write_proposer_transcript",
]
