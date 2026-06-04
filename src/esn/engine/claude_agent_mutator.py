# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Claude-agent-backed mutator (PR A, phase 1.2).

``ClaudeAgentMutator`` is an alternative mutator backend that delegates code
mutation to a Claude agent loop via a ``ClaudeAgentClient``. It is distinct
from ``LLMMutator``: there is no ``mutator_policy`` parameter here — the
class itself is the policy. The isolation boundary is enforced by packing
only whitelisted fields into a ``MutatorInputBundle`` (see
``src/esn/engine/claude_agent_client.py``).

This module intentionally does NOT import ``claude_agent_sdk`` or the
concrete ``ClaudeAgentSDKClient``; it depends only on the ``ClaudeAgentClient``
Protocol so unit tests can inject fakes.
"""

from __future__ import annotations

from typing import Any

from esn.engine.claude_agent_client import (
    ClaudeAgentClient,
    ClaudeAgentClientError,
    MutatorInputBundle,
)
from esn.engine.compiler import validate_program_ast
from esn.engine.domain import DomainSpec
from esn.engine.models import MutationContext, MutationResult
from esn.engine.protocols import ProgramObject


_MUTATOR_POLICY = "claude_agent_sdk"
_AGENT_BACKEND = "claude_agent_sdk"


def _base_metadata(
    *,
    style: str,
    parents: list[ProgramObject],
    context: MutationContext | None,
) -> dict[str, Any]:
    """Build the metadata dict shared by every return path.

    Always populates: ``style``, ``parent_count``, ``mutator_policy``,
    ``agent_backend``, ``agent_used_research``, plus the safe-on-failure
    subset of context fields (``targeted_hypotheses``, ``intended_effect``)
    when a context is available.
    """
    metadata: dict[str, Any] = {
        "style": style,
        "parent_count": len(parents),
        "mutator_policy": _MUTATOR_POLICY,
        "agent_backend": _AGENT_BACKEND,
        "agent_used_research": False,
        # ``research_summary`` is ALWAYS present in the metadata contract
        # (possibly ""), so downstream consumers never have to probe with
        # ``in metadata``. Success and failure-with-partial paths overwrite
        # this with the response's actual value; pure-error paths leave "".
        "research_summary": "",
    }
    if context is not None:
        metadata["targeted_hypotheses"] = list(context.targeted_hypothesis_ids or [])
        metadata["intended_effect"] = context.intended_effect or ""
    return metadata


def _build_bundle(
    parents: list[ProgramObject],
    style: str,
    context: MutationContext,
    domain: DomainSpec,
) -> MutatorInputBundle:
    """Project the mutation inputs onto the isolation-boundary whitelist.

    Only the 13 fields on ``MutatorInputBundle`` may cross into the agent.
    We pull ``parents[0].code`` and nothing else from the parent — no
    ``summary()`` / ``structural_hash()`` / ``serialize()`` calls, no reads
    beyond index 0 of ``parents``.

    Hypotheses are scrubbed to ``{id, family, summary}`` with empty-string
    defaults so scores, diagnostics, and other metadata never leak.
    """
    # Scrub hypothesis entries down to exactly three string fields.
    top_hypotheses_summary: list[dict[str, str]] = []
    for hyp in context.top_hypotheses:
        entry = {
            "id": str(hyp.get("id", "") or ""),
            "family": str(hyp.get("family", "") or ""),
            "summary": str(hyp.get("summary", "") or ""),
        }
        top_hypotheses_summary.append(entry)

    allowed_imports_list = sorted(domain.allowed_imports or frozenset())

    # Mirror ``LLMMutator``'s gating at ``src/esn/engine/mutator.py:479-480``:
    # ``if context.spectral_guidance`` skips the prompt line for an empty
    # dict, so we only propagate a populated guidance payload. Non-empty
    # dicts are str()-ified exactly the way the f-string in the LLMMutator
    # path renders them — parity, not reformatting.
    spectral_guidance_text: str | None = (
        str(context.spectral_guidance) if context.spectral_guidance else None
    )

    return MutatorInputBundle(
        domain_name=domain.name,
        domain_description=domain.description,
        hard_constraints=list(domain.hard_constraints or []),
        allowed_imports=list(allowed_imports_list),
        max_code_lines=domain.max_code_lines,
        program_interface=domain.program_interface,
        parent_code=parents[0].code,
        style=style,
        intended_effect=context.intended_effect or "",
        targeted_hypothesis_ids=list(context.targeted_hypothesis_ids or []),
        top_hypotheses_summary=top_hypotheses_summary,
        mutation_style=context.mutation_style or "",
        search_mode=context.search_mode or "",
        preferred_solution_shape=domain.preferred_solution_shape,
        spectral_guidance=spectral_guidance_text,
    )


class ClaudeAgentMutator:
    """Mutator backend that delegates code mutation to a Claude agent loop.

    Distinct from ``LLMMutator``: uses a bundle-based isolation boundary and
    an agent-loop client (see ``src/esn/engine/claude_agent_client.py``). This
    class is itself the policy; there is no ``mutator_policy`` parameter.
    """

    def __init__(self, client: ClaudeAgentClient, domain: DomainSpec) -> None:
        self._client = client
        self._domain = domain

    def mutate(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult:
        # Empty-parent guard: no context-derived fields, only the base set.
        if not parents:
            return MutationResult(
                code="",
                success=False,
                errors=["no parents provided"],
                metadata=_base_metadata(style=style, parents=parents, context=None),
            )

        # Build the isolation-bounded bundle from the whitelisted inputs only.
        bundle = _build_bundle(parents, style, context, self._domain)

        # Call the agent. Any client-level failure becomes a failed MutationResult.
        try:
            response = self._client.run_mutation(bundle)
        except ClaudeAgentClientError as exc:
            metadata = _base_metadata(style=style, parents=parents, context=context)
            # When parsing failed after ``_run()`` completed, the SDK has
            # already spent turns/tokens — attach the real accounting so
            # single_shot vs claude_agent_sdk comparisons are not biased
            # downward on bad attempts. Mid-stream failures leave
            # ``partial_response=None`` and we write no accounting keys.
            partial = getattr(exc, "partial_response", None)
            if partial is not None:
                metadata["mutation_model"] = partial.model
                metadata["agent_turn_count"] = partial.agent_turn_count
                metadata["input_tokens"] = partial.input_tokens
                metadata["output_tokens"] = partial.output_tokens
                metadata["total_cost_usd"] = partial.total_cost_usd
                metadata["mutator_raw_response"] = partial.raw_response_text
                metadata["research_summary"] = getattr(partial, "research_summary", "")
            return MutationResult(
                code="",
                success=False,
                errors=[f"claude agent client error: {exc}"],
                metadata=metadata,
            )

        # Validate the returned code against the domain's AST constraints.
        validation_errors = validate_program_ast(
            response.code,
            max_lines=self._domain.max_code_lines,
            allowed_imports=self._domain.allowed_imports,
        )
        if validation_errors:
            metadata = _base_metadata(style=style, parents=parents, context=context)
            metadata["mutation_model"] = response.model
            metadata["agent_turn_count"] = response.agent_turn_count
            metadata["input_tokens"] = response.input_tokens
            metadata["output_tokens"] = response.output_tokens
            metadata["total_cost_usd"] = response.total_cost_usd
            metadata["mutator_raw_response"] = response.raw_response_text
            metadata["research_summary"] = response.research_summary
            return MutationResult(
                code="",
                success=False,
                errors=list(validation_errors),
                metadata=metadata,
            )

        # Success: enrich metadata with agent-only fields.
        #
        # Accounting surface (Wave 1): ``mutation_model`` / ``agent_turn_count``
        # / ``input_tokens`` / ``output_tokens`` / ``total_cost_usd`` /
        # ``mutator_raw_response`` are passed through from the Claude Agent
        # SDK's ``ResultMessage`` verbatim. Fields the SDK does not expose
        # are left as ``None`` — we never invent data. Note that on the
        # Claude path ``agent_turn_count`` is the harness-visible proxy for
        # ``mutator_calls`` (semantic overload documented in Wave 2 notes).
        metadata = _base_metadata(style=style, parents=parents, context=context)
        metadata["agent_turn_count"] = response.agent_turn_count
        metadata["agent_summary"] = response.agent_summary
        metadata["diff_summary"] = response.diff_summary
        metadata["mutation_model"] = response.model
        metadata["input_tokens"] = response.input_tokens
        metadata["output_tokens"] = response.output_tokens
        metadata["total_cost_usd"] = response.total_cost_usd
        metadata["mutator_raw_response"] = response.raw_response_text
        metadata["research_summary"] = response.research_summary
        return MutationResult(
            code=response.code,
            success=True,
            errors=[],
            metadata=metadata,
        )


__all__ = ["ClaudeAgentMutator", "_build_bundle"]
