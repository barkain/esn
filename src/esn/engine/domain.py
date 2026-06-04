# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Domain specification for ESN engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from esn.core.models import EvaluationResult  # type: ignore[import-not-found]

from esn.engine.protocols import ProgramCompiler  # type: ignore[import-not-found]


@dataclass
class DomainSpec:
    """Everything the engine needs from a domain.

    The engine is fully domain-agnostic. All domain knowledge lives here.
    """

    name: str
    description: str

    # Initial program
    initial_code: str

    # Compilation
    compiler: ProgramCompiler

    # Evaluation
    evaluator: Callable[[Any], EvaluationResult]

    # Prompt-steering + mutator-side AST guards (with defaults).
    #
    # NOTE: the engine itself never reads these — they are consumed by the
    # mutators (``LLMMutator`` / ``ClaudeAgentMutator``): both render them into
    # the proposal prompt AND feed them to ``validate_program_ast`` so a
    # mutated candidate that violates them is retried before it ever reaches the
    # compiler. The compiler (e.g. ``PythonSandboxCompiler(allowed_imports=...)``)
    # owns the *binding* enforcement at compile time; set it there too if you
    # need a hard guarantee, since a mutator is free to be swapped out.
    allowed_imports: frozenset[str] | None = frozenset()
    max_code_lines: int | None = None

    # Program interface: "solve" (define a solve() function) or "stdio" (runnable stdin/stdout script)
    program_interface: str = "solve"

    # Domain context for LLM prompts
    hard_constraints: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    # Optional domain-specific style instruction overrides.
    # Maps style_name -> instruction string. Falls back to generic defaults.
    style_overrides: dict[str, str] = field(default_factory=dict)

    # Optional advisory prompt-steering toward the preferred shape of a
    # solution for this domain (e.g., "prefer direct constructive solutions
    # over long-running global search"). Rendered verbatim by both the
    # ``LLMMutator`` and ``ClaudeAgentSDKClient`` prompt surfaces as a
    # ``# Preferred solution shape`` section; a ``None`` value renders the
    # fallback string ``(no domain-specific preference)``. Purely advisory:
    # no validator / evaluator / compile effect.
    preferred_solution_shape: str | None = None

    def __post_init__(self) -> None:
        """Catch the most common onboarding mistake: ``initial_code`` that does
        not match the declared ``program_interface``.

        Fails at construction with a message naming ``program_interface`` rather
        than letting the mismatch surface later as an opaque compile/eval error.
        """
        if self.program_interface == "solve" and "def solve(" not in self.initial_code:
            raise ValueError(
                "DomainSpec.program_interface='solve' requires initial_code to "
                "define a `def solve(...)` function (none found). Define one, or "
                "set program_interface='stdio' for a stdin/stdout script."
            )
        if self.program_interface == "stdio" and "def solve(" in self.initial_code:
            raise ValueError(
                "DomainSpec.program_interface='stdio' expects a runnable "
                "stdin/stdout script, but initial_code defines `def solve(...)`. "
                "Set program_interface='solve' if that is the intended interface."
            )
