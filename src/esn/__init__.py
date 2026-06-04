# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""ESN: Epistemic Spectral Novelty Framework.

Public, dependency-light entrypoint. Heavy and optional dependencies
(embeddings, agent SDKs, persistence backends) are imported lazily on
first attribute access via PEP 562 ``__getattr__`` so that ``import esn``
never eagerly pulls torch / chromadb / sentence_transformers / openai /
anthropic / claude_agent_sdk.

    >>> from esn import run, DomainSpec, ESNEngine, LLMMutator
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "CompilerResult",
    "DomainSpec",
    "ESNEngine",
    "EvaluationDiagnostics",
    "EvaluationResult",
    "LLMMutator",
    "MockMutator",
    "PythonSandboxCompiler",
    "StdioCompiler",
    "UvSandboxCompiler",
    "make_agent_analyzer",
    "make_agent_mutator",
    "make_agent_predictor",
    "make_analyzer",
    "make_llm_mutator",
    "make_predictor",
    "run",
]

if TYPE_CHECKING:  # imports for type checkers only — never executed at runtime
    from esn.api import (
        MockMutator,
        make_agent_analyzer,
        make_agent_mutator,
        make_agent_predictor,
        make_analyzer,
        make_llm_mutator,
        make_predictor,
        run,
    )
    from esn.core.models import CompilerResult, EvaluationDiagnostics, EvaluationResult
    from esn.engine.compiler import PythonSandboxCompiler
    from esn.engine.domain import DomainSpec
    from esn.engine.engine import ESNEngine
    from esn.engine.mutator import LLMMutator
    from esn.engine.stdio_compiler import StdioCompiler
    from esn.engine.uv_compiler import UvSandboxCompiler


def __getattr__(name: str) -> object:
    """Lazily resolve public symbols (PEP 562).

    Each branch imports only the module that defines the requested symbol,
    so no heavy/optional dependency is loaded until it is actually needed.
    """
    if name == "run":
        from esn.api import run

        return run
    if name == "make_llm_mutator":
        from esn.api import make_llm_mutator

        return make_llm_mutator
    if name == "make_agent_mutator":
        from esn.api import make_agent_mutator

        return make_agent_mutator
    if name == "make_agent_analyzer":
        from esn.api import make_agent_analyzer

        return make_agent_analyzer
    if name == "make_agent_predictor":
        from esn.api import make_agent_predictor

        return make_agent_predictor
    if name == "make_analyzer":
        from esn.api import make_analyzer

        return make_analyzer
    if name == "make_predictor":
        from esn.api import make_predictor

        return make_predictor
    if name == "DomainSpec":
        from esn.engine.domain import DomainSpec

        return DomainSpec
    if name == "ESNEngine":
        from esn.engine.engine import ESNEngine

        return ESNEngine
    if name == "LLMMutator":
        from esn.engine.mutator import LLMMutator

        return LLMMutator
    if name == "MockMutator":
        from esn.api import MockMutator

        return MockMutator
    if name == "EvaluationResult":
        from esn.core.models import EvaluationResult

        return EvaluationResult
    if name == "EvaluationDiagnostics":
        from esn.core.models import EvaluationDiagnostics

        return EvaluationDiagnostics
    if name == "CompilerResult":
        from esn.core.models import CompilerResult

        return CompilerResult
    if name == "UvSandboxCompiler":
        from esn.engine.uv_compiler import UvSandboxCompiler

        return UvSandboxCompiler
    if name == "PythonSandboxCompiler":
        from esn.engine.compiler import PythonSandboxCompiler

        return PythonSandboxCompiler
    if name == "StdioCompiler":
        from esn.engine.stdio_compiler import StdioCompiler

        return StdioCompiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
