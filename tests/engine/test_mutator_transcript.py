# ruff: noqa: S101
"""Tests for the proposer-turn transcript writer.

These tests exercise the writer as a pure function — no SDK, no network,
no benchmark runner. They cover:

1. A turn with one ``WebSearch`` tool_use + matching tool_result + a
   final text event produces the expected sections/headers.
2. A turn with no tool events (``--mutator-tools none`` path) produces
   only the ``## Final response`` section.
3. A turn with a ``WebFetch`` tool_use + matching tool_result produces
   the ``**URL**`` / ``**Prompt**`` header lines.
4. Oversized tool_result payloads are truncated; small ones are kept
   verbatim.
5. Generic (non-web) tool_result bodies over 1 KB are byte-truncated
   with an explicit marker.
"""

from __future__ import annotations

import json
from pathlib import Path

from esn.engine.mutator_transcript import (
    ProposerTurnEvent,
    summarize_tool_result_content,
    write_proposer_transcript,
)


def _read(path: Path) -> str:
    return path.read_text()


# ---------------------------------------------------------------------------
# Writer: shape of the transcript
# ---------------------------------------------------------------------------


def test_writer_produces_expected_sections_for_research_search(tmp_path: Path) -> None:
    """One WebSearch round-trip + final text => header + turn events + response."""
    events = [
        ProposerTurnEvent(
            kind="tool_use",
            tool_name="WebSearch",
            tool_use_id="toolu_1",
            tool_input={"query": "Strassen matrix multiplication"},
        ),
        ProposerTurnEvent(
            kind="tool_result",
            tool_use_id="toolu_1",
            tool_result_content=[
                {
                    "title": "Strassen algorithm - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Strassen_algorithm",
                    "snippet": "An algorithm for matrix multiplication...",
                },
            ],
        ),
        ProposerTurnEvent(kind="text", text='```json\n{"code": "..."}\n```'),
    ]

    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=3,
        variant_id=1234,
        turn_events=events,
        final_response_text='```json\n{"code": "..."}\n```',
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )

    assert path.exists()
    assert path.name == "gen_03_t1234_proposer.md"
    assert path.parent.name == "transcripts"
    body = _read(path)

    # Header
    assert "# Proposer turn — gen 03 / variant 1234" in body
    assert "**Mutator mode**: agentic mutation (research-enabled)" in body
    assert "**Backend**: claude_agent_sdk" in body
    assert "**Model**: claude-haiku-4-5-20251001" in body
    assert "**Research tools available**: WebSearch, WebFetch" in body
    assert "**Tool invocations**: 1 WebSearch" in body

    # Turn events
    assert "## Turn events" in body
    assert "### Tool use #1 — WebSearch" in body
    assert '**Query**: "Strassen matrix multiplication"' in body
    assert "### Tool result #1" in body

    # Final response
    assert "## Final response (parsed into program)" in body
    assert '{"code": "..."}' in body


def test_writer_no_tool_events_renders_only_final_response(tmp_path: Path) -> None:
    """``--mutator-tools none`` path: no tool_use sections are emitted."""
    events = [ProposerTurnEvent(kind="text", text='{"code": "..."}')]

    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=0,
        variant_id="abc",
        turn_events=events,
        final_response_text='{"code": "..."}',
        mutator_tools="none",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )

    body = _read(path)
    assert path.name == "gen_00_tabc_proposer.md"
    assert "**Mutator mode**: agentic mutation (no tools)" in body
    assert "**Research tools available**: none" in body
    assert "**Tool invocations**: none" in body
    # The turn-events section must NOT appear on the zero-tool path.
    assert "## Turn events" not in body
    assert "### Tool use" not in body
    assert "### Tool result" not in body
    # No-tools mode must NOT render the research-synthesis section.
    assert "## Research synthesis" not in body
    # Final response is always emitted.
    assert "## Final response (parsed into program)" in body
    assert '{"code": "..."}' in body


def test_writer_web_fetch_renders_url_and_prompt_headers(tmp_path: Path) -> None:
    events = [
        ProposerTurnEvent(
            kind="tool_use",
            tool_name="WebFetch",
            tool_use_id="toolu_x",
            tool_input={
                "url": "https://example.com/paper.html",
                "prompt": "Extract the abstract",
            },
        ),
        ProposerTurnEvent(
            kind="tool_result",
            tool_use_id="toolu_x",
            tool_result_content={
                "url": "https://example.com/paper.html",
                "status": 200,
                "text": "Abstract text here.",
            },
        ),
        ProposerTurnEvent(kind="text", text="final"),
    ]
    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=7,
        variant_id=42,
        turn_events=events,
        final_response_text="final",
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )
    body = _read(path)
    assert "### Tool use #1 — WebFetch" in body
    assert "**URL**: https://example.com/paper.html" in body
    assert '**Prompt**: "Extract the abstract"' in body
    assert "**Tool invocations**: 1 WebFetch" in body


def test_writer_header_uses_canonical_framing_labels(tmp_path: Path) -> None:
    """Regression: the transcript header must use the canonical labels
    (``agentic mutation (no tools)`` / ``agentic mutation (research-enabled)``),
    matching the runner console + ``report.md`` header.

    Earlier the writer interpolated ``mutator_tools`` verbatim, producing
    ``agentic mutation (none)`` / ``agentic mutation (research)`` — which
    drifted from the rest of the framing surface. See PR #40 review.
    """
    events = [ProposerTurnEvent(kind="text", text="ok")]

    none_path = write_proposer_transcript(
        run_dir=tmp_path / "none_run",
        generation=0,
        variant_id=1,
        turn_events=events,
        final_response_text="ok",
        mutator_tools="none",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )
    none_body = _read(none_path)
    assert "**Mutator mode**: agentic mutation (no tools)" in none_body
    # The drifted labels must not reappear.
    assert "agentic mutation (none)" not in none_body

    research_path = write_proposer_transcript(
        run_dir=tmp_path / "research_run",
        generation=0,
        variant_id=1,
        turn_events=events,
        final_response_text="ok",
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )
    research_body = _read(research_path)
    assert "**Mutator mode**: agentic mutation (research-enabled)" in research_body
    assert "agentic mutation (research)\n" not in research_body


def test_writer_counts_tool_invocations_by_name(tmp_path: Path) -> None:
    events = [
        ProposerTurnEvent(
            kind="tool_use",
            tool_name="WebSearch",
            tool_use_id="a",
            tool_input={"query": "q1"},
        ),
        ProposerTurnEvent(kind="tool_result", tool_use_id="a", tool_result_content=[]),
        ProposerTurnEvent(
            kind="tool_use",
            tool_name="WebSearch",
            tool_use_id="b",
            tool_input={"query": "q2"},
        ),
        ProposerTurnEvent(kind="tool_result", tool_use_id="b", tool_result_content=[]),
        ProposerTurnEvent(
            kind="tool_use",
            tool_name="WebFetch",
            tool_use_id="c",
            tool_input={"url": "u"},
        ),
        ProposerTurnEvent(kind="tool_result", tool_use_id="c", tool_result_content=""),
        ProposerTurnEvent(kind="text", text="ok"),
    ]
    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=1,
        variant_id=0,
        turn_events=events,
        final_response_text="ok",
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
    )
    body = _read(path)
    # Sorted alphabetical: WebFetch first.
    assert "**Tool invocations**: 1 WebFetch, 2 WebSearch" in body


# ---------------------------------------------------------------------------
# Research-synthesis section (research-enabled mode only)
# ---------------------------------------------------------------------------


def test_writer_renders_research_synthesis_section_in_research_mode(
    tmp_path: Path,
) -> None:
    """research mode + non-empty ``research_summary`` => ``## Research synthesis``.

    The section sits between turn events and the final response so a
    reader can see the agent's synthesis without parsing the final JSON.
    """
    events = [ProposerTurnEvent(kind="text", text="ok")]
    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=4,
        variant_id="r1",
        turn_events=events,
        final_response_text="ok",
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
        research_summary=("Found Strassen; adopted 2-level recursion; rejected Winograd."),
    )
    body = _read(path)
    assert "## Research synthesis" in body
    assert "Found Strassen; adopted 2-level recursion; rejected Winograd." in body


def test_writer_research_synthesis_empty_shows_no_research_used_note(
    tmp_path: Path,
) -> None:
    """research mode + empty summary => section still rendered with a marker.

    This distinguishes "research mode was on but the agent did not use
    research" from "section absent because not in research mode".
    """
    events = [ProposerTurnEvent(kind="text", text="ok")]
    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=4,
        variant_id="r2",
        turn_events=events,
        final_response_text="ok",
        mutator_tools="research",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
        research_summary="",
    )
    body = _read(path)
    assert "## Research synthesis" in body
    assert "_(agent reported no research used)_" in body


def test_writer_research_synthesis_omitted_in_no_tools_mode(
    tmp_path: Path,
) -> None:
    """no-tools mode must NOT render ``## Research synthesis`` even if a
    ``research_summary`` is passed.

    The section is research-mode-only by contract; rendering it in
    no-tools mode would drift from the research-enabled mutation design.
    """
    events = [ProposerTurnEvent(kind="text", text="ok")]
    path = write_proposer_transcript(
        run_dir=tmp_path,
        generation=4,
        variant_id="r3",
        turn_events=events,
        final_response_text="ok",
        mutator_tools="none",
        backend="claude_agent_sdk",
        model="claude-haiku-4-5-20251001",
        research_summary="should not appear",
    )
    body = _read(path)
    assert "## Research synthesis" not in body
    assert "should not appear" not in body


# ---------------------------------------------------------------------------
# Summarization rules
# ---------------------------------------------------------------------------


def test_summarize_generic_keeps_small_content_verbatim() -> None:
    small = "hello world"
    out = summarize_tool_result_content("UnknownTool", small)
    assert out == "hello world"
    assert "truncated" not in out


def test_summarize_generic_truncates_large_content_with_marker() -> None:
    # 2 KB of 'a' exceeds the 1 KB generic cap.
    oversized = "a" * 2048
    out = summarize_tool_result_content("UnknownTool", oversized)
    assert "truncated" in out
    # The kept prefix is <= 1024 bytes.
    assert len(out) < len(oversized)
    assert out.startswith("a" * 128)  # prefix preserved


def test_summarize_web_search_keeps_small_payload_verbatim() -> None:
    content = [
        {"title": "t1", "url": "https://a", "snippet": "s1"},
        {"title": "t2", "url": "https://b", "snippet": "s2"},
    ]
    out = summarize_tool_result_content("WebSearch", content)
    # Small payload: JSON-rendered verbatim, no truncation note.
    assert '"title": "t1"' in out
    assert '"title": "t2"' in out
    assert "_truncated" not in out


def test_summarize_web_search_truncates_many_results_to_first_10() -> None:
    # 50 results, each with a long snippet => way over the 2 KB budget.
    content = [
        {
            "title": f"title{i}",
            "url": f"https://example.com/{i}",
            "snippet": "x" * 500,
        }
        for i in range(50)
    ]
    out = summarize_tool_result_content("WebSearch", content)
    parsed = json.loads(out)
    assert parsed["_truncated"] is True
    assert len(parsed["results"]) == 10
    # Snippets capped at 200 chars.
    for entry in parsed["results"]:
        assert len(entry["snippet"]) <= 200


def test_summarize_web_fetch_truncates_text_with_marker() -> None:
    content = {
        "url": "https://example.com",
        "status": 200,
        "text": "x" * 5000,
    }
    out = summarize_tool_result_content("WebFetch", content)
    parsed = json.loads(out)
    assert parsed["_truncated"] is True
    assert len(parsed["text"]) == 2000
    assert parsed["url"] == "https://example.com"
    assert parsed["status"] == 200


def test_summarize_web_fetch_small_dict_kept_verbatim() -> None:
    content = {"url": "https://example.com", "status": 200, "text": "short"}
    out = summarize_tool_result_content("WebFetch", content)
    parsed = json.loads(out)
    assert parsed == content
    assert "_truncated" not in parsed


def test_summarize_web_fetch_plain_string_truncates_with_marker() -> None:
    content = "y" * 5000
    out = summarize_tool_result_content("WebFetch", content)
    assert out.startswith("y" * 2000)
    assert "truncated" in out
