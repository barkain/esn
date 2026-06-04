# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Presentation-layer framing for the mutator regime.

User-facing output
(console headers, ``report.md`` headers) should describe the
``claude_agent_sdk`` path as "agentic mutation" rather than leaking the
backend-specific identifier. The exact backend and model remain visible
in metadata — this module just centralizes the vocabulary so each
benchmark runner stays in sync.

Tool policy is surfaced via the ``tools`` parameter. Today:

* ``"none"``     -> ``agentic mutation (no tools)``
* ``"research"`` -> ``agentic mutation (research-enabled)``

"research" is backend-agnostic: the specific retrieval tools it resolves
to are decided inside ``ClaudeAgentSDKClient`` and do not leak into the
public framing vocabulary.

Scope: pure string / dict helpers. No I/O, no SDK imports. Safe to
import everywhere, including from tests that don't have the Claude
Agent SDK installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "mutator_mode_label",
    "console_framing_lines",
    "report_framing_lines",
    "run_metadata_fields",
]


def mutator_mode_label(mutator_policy: str, *, tools: str = "none") -> str:
    """Return the public "Mutator mode" label for a given policy.

    ``claude_agent_sdk`` is reframed as ``"agentic mutation (...)"``;
    other policies keep their existing internal-name label so
    single_shot / agentic_v1 consoles are untouched.
    """
    if mutator_policy == "claude_agent_sdk":
        if tools == "research":
            return "agentic mutation (research-enabled)"
        return "agentic mutation (no tools)"
    # For non-claude policies we echo the policy string as-is. This
    # keeps existing behavior stable — the reframe is claude-only.
    return mutator_policy


def console_framing_lines(
    mutator_policy: str,
    *,
    claude_model: str | None = None,
    tools: str = "none",
) -> list[str]:
    """Return the three framing lines for ``claude_agent_sdk`` runs.

    Returns an empty list for any other policy — the caller's existing
    per-policy prints are untouched (per PR A scope).
    """
    if mutator_policy != "claude_agent_sdk":
        return []
    model = claude_model or "<unknown>"
    return [
        f"Mutator mode: {mutator_mode_label(mutator_policy, tools=tools)}",
        "Mutator backend: claude_agent_sdk",
        f"Mutator model: {model}",
    ]


def report_framing_lines(
    mutator_policy: str,
    *,
    mutation_model: str,
    claude_model: str | None = None,
    tools: str = "none",
) -> list[str]:
    """Return the report.md header block for the mutator.

    - ``claude_agent_sdk``: three lines (Mutator mode / backend / model),
      replacing the old misleading ``Mutation model: gpt-4o`` template
      artifact (see issue #36).
    - ``single_shot`` / ``agentic_v1`` / anything else: keep the existing
      one-line ``**Mutation model**: <mutation_model>`` shape.
    """
    if mutator_policy == "claude_agent_sdk":
        model = claude_model or "<unknown>"
        return [
            f"**Mutator mode**: {mutator_mode_label(mutator_policy, tools=tools)}",
            "**Mutator backend**: claude_agent_sdk",
            f"**Mutator model**: {model}",
        ]
    return [f"**Mutation model**: {mutation_model}"]


def _research_tools_available(mutator_policy: str, tools: str) -> list[str]:
    """Return the concrete research tools exposed in this environment.

    Only the ``claude_agent_sdk`` + ``research`` combination today has
    any concrete tools to list. Non-claude policies and the ``none``
    mode get an empty list. This field documents what the ``research``
    policy resolved to at runtime without baking the specific tool
    names into the public CLI vocabulary.

    Kept in lockstep with ``ClaudeAgentSDKClient._RESEARCH_TOOLS`` —
    tests pin that the CLI mode ``research`` and the client's concrete
    list agree. A future environment swap (e.g. adding agentlib tools)
    touches both places; there's no hidden third source.
    """
    if mutator_policy == "claude_agent_sdk" and tools == "research":
        return ["WebSearch", "WebFetch"]
    return []


def run_metadata_fields(
    mutator_policy: str,
    *,
    mutation_model: str,
    claude_model: str | None = None,
    tools: str = "none",
) -> dict[str, Any]:
    """Return the framing keys for run-level metadata JSON.

    Keys (always present):
      - ``mutator_mode``: ``"agentic mutation (...)"`` on the
        ``claude_agent_sdk`` path; otherwise the policy string.
      - ``mutator_backend``: the policy value as-is.
      - ``mutation_model``: the actual model used — ``claude_model``
        on the claude path, ``mutation_model`` otherwise.
      - ``mutator_tools``: ``"none"`` | ``"research"``.
      - ``research_tools_available``: concrete list of tools the
        research policy resolves to in this environment (e.g.
        ``["WebSearch", "WebFetch"]``). Empty for ``none`` mode and
        for non-claude policies.
    """
    if mutator_policy == "claude_agent_sdk":
        model = claude_model or mutation_model
        mode = mutator_mode_label(mutator_policy, tools=tools)
    else:
        model = mutation_model
        mode = mutator_policy
    return {
        "mutator_mode": mode,
        "mutator_backend": mutator_policy,
        "mutation_model": model,
        "mutator_tools": tools,
        "research_tools_available": _research_tools_available(mutator_policy, tools),
    }
