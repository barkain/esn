# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""ESN engine: program-level search architecture.

Public symbols are exposed lazily via PEP 562 ``__getattr__`` so that
``import esn.engine`` (or ``import esn.engine.engine``) does not eagerly pull the
analyzer / predictor / claude-agent surfaces or any heavy/optional
dependency. Each name resolves to the module that defines it on first
access only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "DomainSpec",
    "ESNEngine",
    "LLMMutator",
    "PythonSandboxCompiler",
    "StdioCompiler",
    "UvSandboxCompiler",
    # agentic / LLM surfaces
    "ClaudeAgentMutator",
    "MutatorInputBundle",
    "LLMAnalyzer",
    "LLMPredictor",
    # protocols
    "Analyzer",
    "Mutator",
    "Predictor",
    "ProgramCompiler",
    "ProgramObject",
]

if TYPE_CHECKING:  # type-checker-only imports; never run at import time
    from esn.engine.analyzer import LLMAnalyzer
    from esn.engine.claude_agent_client import MutatorInputBundle
    from esn.engine.claude_agent_mutator import ClaudeAgentMutator
    from esn.engine.compiler import PythonSandboxCompiler
    from esn.engine.domain import DomainSpec
    from esn.engine.engine import ESNEngine
    from esn.engine.mutator import LLMMutator
    from esn.engine.predictor import LLMPredictor
    from esn.engine.protocols import (
        Analyzer,
        Mutator,
        Predictor,
        ProgramCompiler,
        ProgramObject,
    )
    from esn.engine.stdio_compiler import StdioCompiler
    from esn.engine.uv_compiler import UvSandboxCompiler


# Map each public name to the submodule that defines it. The submodule is
# imported on demand, keeping package import free of heavy/optional deps.
_LAZY: dict[str, str] = {
    "ESNEngine": "esn.engine.engine",
    "DomainSpec": "esn.engine.domain",
    "LLMMutator": "esn.engine.mutator",
    "PythonSandboxCompiler": "esn.engine.compiler",
    "StdioCompiler": "esn.engine.stdio_compiler",
    "UvSandboxCompiler": "esn.engine.uv_compiler",
    "ClaudeAgentMutator": "esn.engine.claude_agent_mutator",
    "MutatorInputBundle": "esn.engine.claude_agent_client",
    "LLMAnalyzer": "esn.engine.analyzer",
    "LLMPredictor": "esn.engine.predictor",
    "Analyzer": "esn.engine.protocols",
    "Mutator": "esn.engine.protocols",
    "Predictor": "esn.engine.protocols",
    "ProgramCompiler": "esn.engine.protocols",
    "ProgramObject": "esn.engine.protocols",
}


def __getattr__(name: str) -> object:
    """Lazily resolve a public symbol from its defining submodule (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
