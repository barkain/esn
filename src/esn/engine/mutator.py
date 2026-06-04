# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""LLM-driven program mutator for ESN engine."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from typing import Any, Protocol

from esn.core.llm_adapters import LLMAPIError
from esn.engine.compiler import validate_program_ast
from esn.engine.domain import DomainSpec
from esn.engine.models import MutationContext, MutationResult
from esn.engine.protocols import ProgramObject

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


_STYLE_INSTRUCTIONS = {
    "refine": (
        "Improve the current program. You may adjust parameters, fix bugs, "
        "optimize loops, or make targeted structural changes. Preserve the "
        "overall approach but make it produce better results. Focus on the "
        "specific numerical values, algorithmic details, or data structure "
        "choices that most affect the output quality."
    ),
    "explore": (
        "Try a different constructive strategy for building the solution. "
        "You MUST start from a valid, feasible configuration — never initialize "
        "from random positions. Build a valid solution first, THEN optimize it. "
        "The parent program shows one working approach — find a qualitatively "
        "different one that also works, then improve it."
    ),
    "repair": (
        "The current program has issues. Fix the specific problems: constraint "
        "violations, runtime errors, numerical issues, or output invalidity. "
        "Preserve working logic and fix only what's broken. If the program "
        "crashes, identify the exact failure point and correct it."
    ),
    "radical": (
        "Write a completely different solver, but respect this critical "
        "constraint: your program MUST produce a valid solution. Start by "
        "constructing a feasible configuration using a deterministic or "
        "structured method (not random initialization). Then apply local "
        "optimization (greedy improvement, iterative refinement). "
        "Never use pure random search or any approach that hopes to stumble "
        "onto validity. Validity first, quality second."
    ),
    "synthesize": "Combine the strongest ideas from the provided parent programs into one coherent solver.",
    "recombine": (
        "You are given two parent programs from different search branches. "
        "Produce ONE complete, valid solver that recombines them:\n"
        "1. Preserve correctness — the output MUST be a feasible solution.\n"
        "2. Combine exactly one concrete strength from Parent A and one concrete "
        "strength from Parent B (e.g. A's initialization with B's optimizer, or "
        "A's data structure with B's refinement loop).\n"
        "3. Do not copy both parents verbatim; synthesize a single coherent program.\n"
        "4. Briefly explain the intended merge in the JSON metadata only, not "
        "as comments in the code."
    ),
}


_VALID_POLICIES = ("single_shot", "agentic_v1")


_UNICODE_REPLACEMENTS = {
    "\u00d7": "*",  # ×
    "\u00f7": "/",  # ÷
    "\u221a": "**0.5",  # √ (approximate — won't handle √(expr) correctly but better than rejection)
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u2260": "!=",  # ≠
    "\u00b1": "+/-",  # ± (will still fail but at least flagged)
    "\u2192": "->",  # →
    "\u2190": "<-",  # ←
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u2013": "-",  # – en-dash
    "\u2014": "-",  # — em-dash
    "\u2010": "-",  # ‐ hyphen
    "\u2212": "-",  # − minus sign
}


def _runtime_budget_hint(allowed_imports: frozenset[str] | None) -> str:
    """Render the wall-clock runtime-budget hint, domain-aware.

    When ``time`` is importable (allowlist is ``None`` or contains
    ``"time"``), the hint concretely points at ``time.time()``. Otherwise,
    render a generic iteration-cap variant that names NO specific stdlib
    module — otherwise the post-mutation AST validator in a restricted
    domain would reject exactly the code the hint induced.
    """
    allows_time = allowed_imports is None or "time" in allowed_imports
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


class LLMMutator:
    """LLM-driven mutation with style-specific prompting and code validation."""

    def __init__(
        self,
        llm_client: LLMClient,
        domain: DomainSpec,
        mutator_policy: str = "single_shot",
    ) -> None:
        self._llm = llm_client
        self._domain = domain
        if mutator_policy not in _VALID_POLICIES:
            raise ValueError(
                f"Unknown mutator_policy={mutator_policy!r}. Valid policies: {_VALID_POLICIES}"
            )
        self._policy = mutator_policy

    # Maximum number of mutator attempts per candidate. With reasoning models
    # like o3, the LLM occasionally returns empty / truncated code when the
    # reasoning-token budget is exhausted, so we retry with explicit feedback.
    # Phase 0.2 RNG/mutator-collapse fix.
    _MAX_ATTEMPTS = 3

    def mutate(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult:
        if self._policy == "single_shot":
            return self._mutate_single_shot(parents, style, context)
        if self._policy == "agentic_v1":
            return self._mutate_agentic_v1(parents, style, context)
        # unreachable — __init__ validates
        raise ValueError(f"Unknown mutator_policy={self._policy!r}")

    def _mutate_single_shot(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult:
        last_error: str = ""
        last_code: str = ""
        last_metadata: dict[str, Any] = {}
        try:
            system_prompt = self._build_system_prompt(style)
            base_user_prompt = self._build_user_prompt(parents, style, context)
            for attempt in range(self._MAX_ATTEMPTS):
                user_prompt = base_user_prompt
                if attempt > 0 and last_error:
                    # Append explicit feedback so the model knows what went wrong.
                    user_prompt = (
                        f"{base_user_prompt}\n\n"
                        f"PREVIOUS ATTEMPT FAILED: {last_error}\n"
                        "Return ONLY the complete Python code for the program. "
                        "Do not include reasoning, explanations, or markdown. "
                        "Make sure the code is syntactically complete and defines "
                        "the required entry point."
                    )
                response = self._llm(system_prompt, user_prompt)
                try:
                    code, metadata = self._parse_response(response)
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    continue
                code = self._sanitize_code(code)
                errors = validate_program_ast(
                    code,
                    max_lines=self._domain.max_code_lines,
                    allowed_imports=self._domain.allowed_imports,
                )
                if errors:
                    last_error = "; ".join(errors)
                    last_code = code
                    last_metadata = metadata
                    continue
                metadata.setdefault("style", style)
                metadata.setdefault("targeted_hypotheses", context.targeted_hypothesis_ids)
                metadata.setdefault("intended_effect", context.intended_effect)
                metadata.setdefault("parent_count", len(parents))
                metadata.setdefault("mutator_attempts", attempt + 1)
                return MutationResult(code=code, success=True, metadata=metadata)
            # All attempts exhausted.
            last_metadata.setdefault("mutator_attempts", self._MAX_ATTEMPTS)
            return MutationResult(
                code=last_code,
                success=False,
                errors=[last_error or "Mutator failed after retries"],
                metadata=last_metadata,
            )
        except LLMAPIError:
            raise  # Fatal API errors must propagate — do not swallow
        except Exception as exc:
            return MutationResult(code="", success=False, errors=[str(exc)])

    def _mutate_agentic_v1(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> MutationResult:
        """3-pass agentic mutation: draft -> critique -> finalize.

        At most 3 LLM calls. No retry within a pass. See design doc D4 for the
        fallback matrix that governs which pass's output is returned when a
        later pass fails.
        """
        # ---- Pass 1: draft ----
        draft_system = self._build_system_prompt(style)
        draft_user = self._build_user_prompt(parents, style, context)
        try:
            draft_response = self._llm(draft_system, draft_user)
        except LLMAPIError:
            raise  # Fatal API errors must propagate — do not swallow

        try:
            draft_code, draft_metadata = self._parse_response(draft_response)
        except Exception as exc:  # noqa: BLE001
            # Draft parse failure — return early with no agentic_* keys.
            return MutationResult(
                code="",
                success=False,
                errors=[str(exc)],
                metadata={},
            )

        draft_code = self._sanitize_code(draft_code)
        if not draft_code:
            return MutationResult(
                code="",
                success=False,
                errors=["Mutator returned empty code"],
                metadata={},
            )

        draft_errors = validate_program_ast(
            draft_code,
            max_lines=self._domain.max_code_lines,
            allowed_imports=self._domain.allowed_imports,
        )
        if draft_errors:
            return MutationResult(
                code=draft_code,
                success=False,
                errors=["; ".join(draft_errors)],
                metadata={},
            )

        draft_diff_summary = draft_metadata.get("diff_summary", "")
        draft_intended_effect = draft_metadata.get("intended_effect", "")

        def _finalize_metadata(
            *,
            pass_count: int,
            fallback: str,
            chosen_metadata: dict[str, Any],
            critique_text: str,
        ) -> dict[str, Any]:
            """Build the metadata dict for a successful agentic_v1 return."""
            metadata: dict[str, Any] = dict(chosen_metadata)
            metadata.setdefault("style", style)
            metadata.setdefault("targeted_hypotheses", context.targeted_hypothesis_ids)
            metadata.setdefault("intended_effect", context.intended_effect)
            metadata.setdefault("parent_count", len(parents))
            metadata.setdefault("mutator_attempts", pass_count)
            metadata["mutator_policy"] = "agentic_v1"
            metadata["agentic_pass_count"] = pass_count
            metadata["agentic_draft_code"] = draft_code
            metadata["agentic_critique"] = critique_text
            metadata["agentic_fallback"] = fallback
            return metadata

        # ---- Pass 2: critique ----
        critique_system = self._build_critique_system_prompt(style)
        critique_user = self._build_critique_user_prompt(
            parents,
            style,
            context,
            draft_code,
            draft_diff_summary,
            draft_intended_effect,
        )
        try:
            critique_response = self._llm(critique_system, critique_user)
        except LLMAPIError:
            raise

        critique_text = self._parse_critique_response(critique_response)
        if not critique_text:
            # Skip finalize — return draft as successful result.
            return MutationResult(
                code=draft_code,
                success=True,
                metadata=_finalize_metadata(
                    pass_count=2,
                    fallback="critique_parse_failed",
                    chosen_metadata=draft_metadata,
                    critique_text="",
                ),
            )

        # ---- Pass 3: finalize ----
        finalize_system = self._build_finalize_system_prompt(style)
        finalize_user = self._build_finalize_user_prompt(
            parents, style, context, draft_code, critique_text
        )
        try:
            finalize_response = self._llm(finalize_system, finalize_user)
        except LLMAPIError:
            raise

        try:
            final_code, final_metadata = self._parse_response(finalize_response)
        except Exception:  # noqa: BLE001
            # Finalize parse failed — return draft.
            return MutationResult(
                code=draft_code,
                success=True,
                metadata=_finalize_metadata(
                    pass_count=3,
                    fallback="finalize_parse_failed",
                    chosen_metadata=draft_metadata,
                    critique_text=critique_text,
                ),
            )

        final_code = self._sanitize_code(final_code)
        final_errors: list[str] = []
        if not final_code:
            final_errors = ["Mutator returned empty code"]
        else:
            final_errors = validate_program_ast(
                final_code,
                max_lines=self._domain.max_code_lines,
                allowed_imports=self._domain.allowed_imports,
            )
        if final_errors:
            # Finalize validation failed — return draft.
            return MutationResult(
                code=draft_code,
                success=True,
                metadata=_finalize_metadata(
                    pass_count=3,
                    fallback="finalize_validation_failed",
                    chosen_metadata=draft_metadata,
                    critique_text=critique_text,
                ),
            )

        return MutationResult(
            code=final_code,
            success=True,
            metadata=_finalize_metadata(
                pass_count=3,
                fallback="none",
                chosen_metadata=final_metadata,
                critique_text=critique_text,
            ),
        )

    def _sanitize_code(self, code: str) -> str:
        """Replace common Unicode characters with ASCII equivalents."""
        for unicode_char, ascii_replacement in _UNICODE_REPLACEMENTS.items():
            if unicode_char in code:
                code = code.replace(unicode_char, ascii_replacement)
        return code

    def _build_system_prompt(self, style: str) -> str:
        instruction = self._domain.style_overrides.get(style) or _STYLE_INSTRUCTIONS.get(
            style, _STYLE_INSTRUCTIONS["refine"]
        )

        if self._domain.program_interface == "stdio":
            interface_instruction = (
                "Write a COMPLETE, SELF-CONTAINED Python script. Requirements:\n"
                "1. Define a main() function containing ALL logic\n"
                "2. End with: if __name__ == '__main__': main()\n"
                "3. Read ALL input from stdin, write ALL output to stdout\n"
                "4. Do NOT use solve() — this runs as a standalone script\n"
                "5. Do NOT use dir(), vars(), getattr(), eval(), exec(), or inspect functions\n"
                "6. Do NOT use threading, multiprocessing, or async — single-threaded only\n"
                "7. Output ONLY the required format — no debug prints, no extra whitespace\n"
                "8. The program must be syntactically complete — no truncated functions or missing closing brackets\n"
            )
        else:
            interface_instruction = (
                "Do NOT include `if __name__ == '__main__':` blocks, test code, or print statements. "
                "Return ONLY the solve() function and any helper functions it needs.\n"
            )

        return (
            f"You are improving a program for the domain '{self._domain.name}'.\n"
            f"Domain description: {self._domain.description}\n"
            "\n"
            "CRITICAL RUNTIME CONSTRAINT: Your program must complete within the time limit "
            "specified in the hard constraints below. "
            "Use bounded loops with small constant factors. Avoid:\n"
            "- Grid searches with more than 200 total evaluations\n"
            "- Nested loops where the inner loop is O(n**2) and runs more than 20 iterations\n"
            "- Multi-phase optimization with more than 3 phases\n"
            "Prefer: direct computation, single-pass refinement, greedy algorithms with early stopping.\n"
            "\n"
            + _runtime_budget_hint(self._domain.allowed_imports)
            + "\n"
            + interface_instruction
            + "\n"
            f"Mutation style: {style}\n"
            f"Instruction: {instruction}\n"
            "CRITICAL: Use ONLY ASCII characters in your code. "
            "Do NOT use Unicode math symbols (\u00d7 \u00f7 \u2264 \u2265 \u00b1 \u2192 \u2260), smart quotes (\u201c \u201d \u2018 \u2019), "
            "en-dashes (\u2013), or any non-ASCII characters. "
            "Use: ** for power, * for multiply, >= <= for comparison, != for not-equal, "
            "' and \" for quotes, - for minus/hyphen.\n"
            "Your code must compile and run without errors. Double-check:\n"
            "- All parentheses, brackets, and braces are matched\n"
            "- All string literals are properly closed\n"
            "- All indentation is consistent (use 4 spaces)\n"
            "\n"
            "Return either raw Python code or JSON with fields "
            '{"code": "...", "diff_summary": "...", "intended_effect": "..."}.\n'
            "Output only the program content / JSON. No markdown explanation.\n"
            + (
                f"Maximum lines: {self._domain.max_code_lines}\n"
                if self._domain.max_code_lines is not None
                else ""
            )
            + (
                f"Allowed imports: {sorted(self._domain.allowed_imports)}\n"
                if self._domain.allowed_imports is not None
                else "Allowed imports: any (unrestricted)\n"
            )
            + f"Hard constraints: {self._domain.hard_constraints}\n"
            + "\n"
            + "# Preferred solution shape\n"
            + f"{self._domain.preferred_solution_shape or '(no domain-specific preference)'}\n"
        )

    def _build_user_prompt(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
    ) -> str:
        parts = [
            f"Search mode: {context.search_mode}",
            f"Search temperature: {context.search_temperature:.2f}",
            f"Intended effect: {context.intended_effect or 'improve the score while preserving validity'}",
        ]
        if context.top_hypotheses:
            cluster_mode = any(h.get("source") == "cluster" for h in context.top_hypotheses)
            header = (
                "Top hypotheses (BBP cluster representatives):"
                if cluster_mode
                else "Top hypotheses:"
            )
            parts.append(header)
            for hyp in context.top_hypotheses[:8]:
                tag = " [cluster]" if hyp.get("source") == "cluster" else ""
                parts.append(f"- [{hyp.get('id', '?')}]{tag} {hyp.get('text', '')}")
        if context.spectral_guidance:
            parts.append(f"Spectral guidance: {context.spectral_guidance}")
        if context.diagnostics:
            parts.append(f"Diagnostics: {context.diagnostics}")
        if style == "radical" and context.score_history and context.score_history.get("best"):
            best = context.score_history["best"]
            parts.append(
                f"The current best score is {best:.4f}. "
                "Your solution should aim to match or exceed this quality."
            )
        if context.error_context:
            parts.append(f"Previous error context: {context.error_context}")
        if self._domain.hints:
            parts.append("Domain hints:")
            for hint in self._domain.hints[:8]:
                parts.append(f"- {hint}")
        if self._domain.examples:
            parts.append("Examples:")
            for example in self._domain.examples[:2]:
                parts.append(example)

        # Family failure context
        if context.family_failure_reasons:
            parts.append("\n## Why Some Families Failed Recently")
            for fam, reasons in context.family_failure_reasons.items():
                # Deduplicate and count occurrences
                reason_counts: dict[str, int] = {}
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                for reason, count in sorted(
                    reason_counts.items(), key=lambda x: x[1], reverse=True
                )[:3]:
                    parts.append(
                        f'- {fam}: "{reason}" ({count} occurrence{"s" if count > 1 else ""})'
                    )

        if style in ("synthesize", "recombine") and len(parents) > 1:
            if style == "recombine":
                parts.append("Parent A (anchor — higher best score):")
                parts.append(parents[0].code)
                parts.append("Parent B (donor — diverse branch):")
                parts.append(parents[1].code)
            else:
                parts.append("Parent programs to combine:")
                for idx, parent in enumerate(parents, start=1):
                    parts.extend([f"Parent {idx}:", parent.code])
        else:
            parts.extend(["Parent program:", parents[0].code])

        # Global search narrative for explore/radical styles
        if style in ("explore", "radical"):
            self._append_search_narrative(parts, style, context, parents=parents)

        return "\n".join(parts)

    def _append_search_narrative(
        self,
        parts: list[str],
        style: str,
        context: MutationContext,
        parents: list[ProgramObject] | None = None,
    ) -> None:
        """Add global search context for explore/radical prompts."""
        # Best program so far (skip if same as parent to avoid duplication)
        if context.best_code:
            parent_is_best = (
                parents
                and len(parents) > 0
                and context.best_code.strip() == parents[0].code.strip()
            )
            if parent_is_best:
                parts.append(
                    f"\nThe parent program IS the current best (score: {context.best_score:.4f})."
                )
            else:
                parts.append(f"\n## Best Program (score: {context.best_score:.4f})")
                parts.append(context.best_code)

        # Recent attempts (with failure reasons when available)
        if context.recent_attempts:
            parts.append("\n## Recent Attempts")
            for attempt in context.recent_attempts:
                mark = "\u2713" if attempt.get("success") else "\u2717"
                line = f"- {attempt.get('style', '?')}: score {attempt.get('score', 0):.4f} {mark}"
                if not attempt.get("success") and attempt.get("error"):
                    line += f" — {attempt['error']}"
                parts.append(line)

        # Family-level summaries (replaces per-program archive strategies)
        if context.family_summaries:
            parts.append("\n## Known Solver Families")
            for summary in context.family_summaries:
                parts.append(f"- {summary}")
        elif context.archive_families:
            # Fallback to archive families if no family tracking yet
            parts.append("\n## Known Strategies in Archive")
            for desc in context.archive_families:
                parts.append(f"- {desc}")

        if context.parent_family:
            parts.append(f"\n## Current Parent Family: {context.parent_family}")

        # Search state
        parts.append("\n## Search State")
        parts.append(f"- Stagnation: {context.stagnation_gens} generations without improvement")
        parts.append(f"- Temperature: {context.search_temperature:.2f}")

        # Style-specific guidance with family awareness
        if style == "explore":
            family_note = ""
            if context.parent_family:
                family_note = (
                    f" The parent uses a {context.parent_family} approach. "
                    "Try a DIFFERENT family -- see the family summaries above "
                    "for what's been tried and what hasn't."
                )
            parts.append(
                "\nThe parent program and best program are shown above. "
                "Study both. Your goal is to find a QUALITATIVELY DIFFERENT "
                "approach that can beat the best score. Do not make minor "
                "variations -- invent a new algorithm family." + family_note
            )

            # Within-family exploit: underdeveloped promising families
            if context.family_summaries:
                self._append_underdeveloped_family_hints(parts, context)

        elif style == "radical":
            parts.append(
                "\nStudy the family summaries. Avoid families that are "
                "plateauing. Try an approach from an untried or underdeveloped "
                "family. Your job is to write a completely new solver that "
                "approaches the problem from a fundamentally different angle "
                "than anything tried so far."
            )

    def _append_underdeveloped_family_hints(
        self,
        parts: list[str],
        context: MutationContext,
    ) -> None:
        """Suggest exploiting within underdeveloped but promising families."""
        for summary in context.family_summaries:
            # Parse summary lines like "hex: best=0.1234, 3 attempts ..."
            try:
                name = summary.split(":")[0].strip()
                if "attempts" not in summary or "best=" not in summary:
                    continue
                best_str = summary.split("best=")[1].split(",")[0]
                best_val = float(best_str)
                attempts_str = summary.split(", ")[1].split(" ")[0]
                attempts = int(attempts_str)
                if best_val > 0 and attempts <= 4:
                    parts.append(
                        f"\nThe {name} family reached {best_val:.4f} in only "
                        f"{attempts} attempts. Consider writing a stronger "
                        "solver in the same family rather than switching to a "
                        "new one."
                    )
            except (ValueError, IndexError):
                continue

    # ------------------------------------------------------------------
    # agentic_v1 helpers: critique + finalize prompt construction
    # ------------------------------------------------------------------

    def _build_critique_system_prompt(self, style: str) -> str:
        """Terse critique system prompt (agentic_v1, per design D6)."""
        del style  # style is embedded in the user prompt; system stays generic
        return (
            "You are reviewing a candidate Python program generated for domain "
            f"'{self._domain.name}'. Return a brief structured critique focused "
            "on: (1) the most likely compile/runtime failure, (2) the most "
            "likely constraint violation, (3) whether the edit is substantive "
            "or cosmetic, (4) the single highest-value correction before "
            "submission. Keep it to 5-10 bullet points. No code. No rewrite. "
            "Just the critique."
        )

    def _build_critique_user_prompt(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
        draft_code: str,
        draft_diff_summary: str,
        draft_intended_effect: str,
    ) -> str:
        """User prompt for the critique pass: mutation block + draft metadata + draft code."""
        mutation_block = self._build_user_prompt(parents, style, context)
        parts = [
            "## Mutation context (for reference)",
            mutation_block,
            "",
            "## Draft candidate metadata",
            f"- diff_summary: {draft_diff_summary or '(none)'}",
            f"- intended_effect: {draft_intended_effect or '(none)'}",
            "",
            "## Draft candidate code",
            draft_code,
            "",
            "Provide your critique now as 5-10 bullet points.",
        ]
        return "\n".join(parts)

    def _build_finalize_system_prompt(self, style: str) -> str:
        """Finalize system prompt (agentic_v1, per design D7)."""
        del style
        return (
            "You are finalizing a Python program for domain "
            f"'{self._domain.name}'. Given a draft and a critique, produce the "
            "final revised candidate. Return JSON: "
            '{"code": "...", "diff_summary": "...", "intended_effect": "..."} '
            "OR raw Python code. Same rules as the draft: ASCII only, validity "
            "first, hard constraints honored."
        )

    def _build_finalize_user_prompt(
        self,
        parents: list[ProgramObject],
        style: str,
        context: MutationContext,
        draft_code: str,
        critique_text: str,
    ) -> str:
        """User prompt for the finalize pass: mutation block + draft + critique."""
        mutation_block = self._build_user_prompt(parents, style, context)
        parts = [
            "## Mutation context (for reference)",
            mutation_block,
            "",
            "## Draft candidate code",
            draft_code,
            "",
            "## Critique of the draft",
            critique_text,
            "",
            "Produce the final revised candidate now. Address the critique's "
            "highest-value correction first. Return JSON "
            '{"code": "...", "diff_summary": "...", "intended_effect": "..."} '
            "or raw Python code.",
        ]
        return "\n".join(parts)

    def _parse_critique_response(self, raw: str) -> str:
        """Strip markdown fences and trim whitespace from a critique response.

        Returns an empty string when the response is empty or whitespace-only
        after cleaning.
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        if not text:
            return ""
        return text

    def _parse_response(self, response: str) -> tuple[str, dict[str, Any]]:
        text = response.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try JSON extraction (o4-mini sometimes wraps code in JSON)
        if text.startswith("{"):
            code, metadata = self._try_extract_json(text)
            if code is not None:
                return code, metadata
            # JSON parse failed — fall through to treat as raw code

        # Some models emit valid raw code followed by a trailing JSON metadata
        # object on a new line. Strip that suffix if it parses cleanly.
        text, trailing_metadata = self._strip_trailing_json_metadata(text)

        if not text:
            raise ValueError("Mutator returned empty code")
        return text, trailing_metadata

    def _try_extract_json(self, text: str) -> tuple[str | None, dict[str, Any]]:
        """Attempt to parse *text* as JSON and pull out the ``code`` field.

        Returns ``(code, metadata)`` on success, ``(None, {})`` when the text
        is not valid JSON or has no usable ``code`` field.
        """
        try:
            data = json.loads(text)
        except JSONDecodeError:
            return None, {}

        if not isinstance(data, dict):
            return None, {}

        code = data.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return None, {}

        # Unescape literal \\n that LLMs sometimes emit inside JSON strings
        if "\\n" in code and "\n" not in code:
            code = code.replace("\\n", "\n")

        metadata = {
            "diff_summary": data.get("diff_summary", ""),
            "intended_effect": data.get("intended_effect", ""),
        }
        return code.strip(), metadata

    def _strip_trailing_json_metadata(self, text: str) -> tuple[str, dict[str, Any]]:
        """Strip a trailing JSON metadata object appended after raw code.

        Expected shape:
          <python code>
          {"code": "...", "diff_summary": "...", "intended_effect": "..."}
        """
        marker = '\n{"code":'
        idx = text.rfind(marker)
        if idx == -1:
            return text, {}

        candidate_json = text[idx + 1 :].strip()
        try:
            data = json.loads(candidate_json)
        except JSONDecodeError:
            return text, {}
        if not isinstance(data, dict):
            return text, {}

        metadata = {
            "diff_summary": data.get("diff_summary", ""),
            "intended_effect": data.get("intended_effect", ""),
        }
        return text[:idx].rstrip(), metadata
