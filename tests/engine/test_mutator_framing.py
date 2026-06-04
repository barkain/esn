# ruff: noqa: S101
"""Framing-layer tests for PR A (agentic-mutation framing cleanup).

Exercises the pure helpers in ``src/esn/engine/mutator_framing.py`` and pins
the user-visible behavior of the agentic-mutation framing:

1. ``claude_agent_sdk`` runs render as "agentic mutation (no tools)" in
   report.md and console output, and carry the Claude model (not
   ``gpt-4o``, which was the stale template artifact from issue #36).
2. ``single_shot`` runs keep the existing one-line ``Mutation model``
   shape — regression protection for pre-existing benchmarks.
3. Run-level metadata always gains the four framing keys
   (``mutator_mode``, ``mutator_backend``, ``mutation_model``,
   ``mutator_tools``).
"""

from __future__ import annotations

from esn.engine.mutator_framing import (
    console_framing_lines,
    mutator_mode_label,
    report_framing_lines,
    run_metadata_fields,
)


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------


def test_label_claude_agent_sdk_no_tools():
    assert mutator_mode_label("claude_agent_sdk") == "agentic mutation (no tools)"


def test_label_claude_agent_sdk_research():
    assert (
        mutator_mode_label("claude_agent_sdk", tools="research")
        == "agentic mutation (research-enabled)"
    )


def test_label_single_shot_echoes_policy():
    assert mutator_mode_label("single_shot") == "single_shot"


def test_label_agentic_v1_echoes_policy():
    assert mutator_mode_label("agentic_v1") == "agentic_v1"


# ---------------------------------------------------------------------------
# Console framing
# ---------------------------------------------------------------------------


def test_console_claude_agent_sdk_block():
    lines = console_framing_lines("claude_agent_sdk", claude_model="claude-haiku-4-5-20251001")
    assert lines == [
        "Mutator mode: agentic mutation (no tools)",
        "Mutator backend: claude_agent_sdk",
        "Mutator model: claude-haiku-4-5-20251001",
    ]


def test_console_single_shot_is_empty():
    # PR A is a claude-only reframe; other policies keep their existing
    # console prints untouched. The helper must return an empty list.
    assert console_framing_lines("single_shot") == []


def test_console_agentic_v1_is_empty():
    assert console_framing_lines("agentic_v1") == []


# ---------------------------------------------------------------------------
# report.md framing
# ---------------------------------------------------------------------------


def test_report_claude_agent_sdk_uses_claude_model():
    lines = report_framing_lines(
        "claude_agent_sdk",
        mutation_model="gpt-4o",  # stale template value; must NOT appear
        claude_model="claude-haiku-4-5-20251001",
    )
    assert lines == [
        "**Mutator mode**: agentic mutation (no tools)",
        "**Mutator backend**: claude_agent_sdk",
        "**Mutator model**: claude-haiku-4-5-20251001",
    ]
    joined = "\n".join(lines)
    assert "agentic mutation (no tools)" in joined
    assert "claude-haiku-4-5-20251001" in joined
    # Primary bug pinned: the stale "Mutation model: gpt-4o" line that
    # issue #36 flagged must NOT appear on the claude path.
    assert "Mutation model: gpt-4o" not in joined
    assert "**Mutation model**: gpt-4o" not in joined


def test_report_single_shot_keeps_legacy_shape():
    # Regression guard: the existing header for single_shot runs is
    # preserved — same key, same formatting — so non-claude benchmark
    # docs don't need retroactive edits.
    lines = report_framing_lines("single_shot", mutation_model="o4-mini", claude_model=None)
    assert lines == ["**Mutation model**: o4-mini"]


def test_report_agentic_v1_keeps_legacy_shape():
    lines = report_framing_lines("agentic_v1", mutation_model="o3", claude_model=None)
    assert lines == ["**Mutation model**: o3"]


# ---------------------------------------------------------------------------
# Run-level metadata
# ---------------------------------------------------------------------------


def test_metadata_claude_agent_sdk_keys_and_values():
    meta = run_metadata_fields(
        "claude_agent_sdk",
        mutation_model="ignored-for-claude",
        claude_model="claude-haiku-4-5-20251001",
    )
    assert set(meta.keys()) == {
        "mutator_mode",
        "mutator_backend",
        "mutation_model",
        "mutator_tools",
        "research_tools_available",
    }
    assert meta["mutator_mode"] == "agentic mutation (no tools)"
    assert meta["mutator_backend"] == "claude_agent_sdk"
    assert meta["mutation_model"] == "claude-haiku-4-5-20251001"
    assert meta["mutator_tools"] == "none"
    assert meta["research_tools_available"] == []


def test_metadata_single_shot_keys_and_values():
    meta = run_metadata_fields("single_shot", mutation_model="o4-mini", claude_model=None)
    assert meta == {
        "mutator_mode": "single_shot",
        "mutator_backend": "single_shot",
        "mutation_model": "o4-mini",
        "mutator_tools": "none",
        "research_tools_available": [],
    }


def test_metadata_agentic_v1_keys_and_values():
    meta = run_metadata_fields("agentic_v1", mutation_model="o3", claude_model=None)
    assert meta == {
        "mutator_mode": "agentic_v1",
        "mutator_backend": "agentic_v1",
        "mutation_model": "o3",
        "mutator_tools": "none",
        "research_tools_available": [],
    }


def test_metadata_always_has_mutator_tools_none_by_default():
    # Default (no ``tools`` kwarg) pins ``mutator_tools == "none"``
    # everywhere; the ``research`` variant requires opting in via the
    # ``--mutator-tools research`` flag.
    for policy in ("claude_agent_sdk", "single_shot", "agentic_v1"):
        meta = run_metadata_fields(
            policy,
            mutation_model="whatever",
            claude_model="claude-haiku-4-5-20251001",
        )
        assert meta["mutator_tools"] == "none"
        assert meta["research_tools_available"] == []


# ---------------------------------------------------------------------------
# research-enabled variants propagate through every framing surface
# ---------------------------------------------------------------------------


def test_console_claude_agent_sdk_research_block():
    lines = console_framing_lines(
        "claude_agent_sdk",
        claude_model="claude-haiku-4-5-20251001",
        tools="research",
    )
    assert lines == [
        "Mutator mode: agentic mutation (research-enabled)",
        "Mutator backend: claude_agent_sdk",
        "Mutator model: claude-haiku-4-5-20251001",
    ]


def test_report_claude_agent_sdk_research_uses_claude_model():
    lines = report_framing_lines(
        "claude_agent_sdk",
        mutation_model="gpt-4o",
        claude_model="claude-haiku-4-5-20251001",
        tools="research",
    )
    assert lines == [
        "**Mutator mode**: agentic mutation (research-enabled)",
        "**Mutator backend**: claude_agent_sdk",
        "**Mutator model**: claude-haiku-4-5-20251001",
    ]


def test_metadata_claude_agent_sdk_research_keys_and_values():
    meta = run_metadata_fields(
        "claude_agent_sdk",
        mutation_model="ignored-for-claude",
        claude_model="claude-haiku-4-5-20251001",
        tools="research",
    )
    assert meta == {
        "mutator_mode": "agentic mutation (research-enabled)",
        "mutator_backend": "claude_agent_sdk",
        "mutation_model": "claude-haiku-4-5-20251001",
        "mutator_tools": "research",
        "research_tools_available": ["WebSearch", "WebFetch"],
    }


def test_research_label_does_not_leak_into_non_claude_policies():
    # Defense-in-depth: even if a caller passes tools="research" to a
    # non-claude policy (shouldn't happen in practice — runners gate
    # the flag on claude_agent_sdk), the label is still the bare policy
    # string. The reframe extends the claude path only.
    assert mutator_mode_label("single_shot", tools="research") == "single_shot"
    assert mutator_mode_label("agentic_v1", tools="research") == "agentic_v1"
    meta = run_metadata_fields(
        "single_shot",
        mutation_model="o4-mini",
        claude_model=None,
        tools="research",
    )
    # Non-claude policies must NOT report research_tools_available.
    assert meta["research_tools_available"] == []
