# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Onboarding-friction guards added to ESN's public surface.

Covers three newcomer-facing fixes:

* :func:`esn.make_agent_analyzer` (and :func:`esn.make_agent_predictor`) — the
  key-FREE, subscription-backed novelty drivers — are re-exported and degrade
  with a clear ``[agent]``-extra error when the SDK is absent.
* The engine guards the ``domain.evaluator`` boundary: a wrong return type
  raises a clear ``ValueError`` instead of an opaque attribute error deep in
  the engine.
* :class:`~esn.engine.domain.DomainSpec` validates ``initial_code`` against the
  declared ``program_interface`` at construction.
"""

from __future__ import annotations

import pytest

import esn
from esn.core.models import CompilerResult
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine

SEED_CODE = "def solve():\n    return [1, 2, 3]\n"


class _AlwaysCompileCompiler:
    """Minimal ProgramCompiler stub: compiles any source to a passthrough artifact."""

    def compile(self, source: str) -> CompilerResult:
        return CompilerResult(artifact=source, success=True)


# ---------------------------------------------------------------------------
# make_agent_analyzer / make_agent_predictor: re-export + [agent]-extra guard
# ---------------------------------------------------------------------------


def test_agent_factories_are_reexported():
    assert callable(esn.make_agent_analyzer)
    assert callable(esn.make_agent_predictor)
    # Public surface advertises them.
    assert "make_agent_analyzer" in esn.__all__
    assert "make_agent_predictor" in esn.__all__


def test_make_agent_analyzer_returns_real_analyzer():
    # Constructing the factory must NOT require the [agent] extra — the SDK is
    # imported lazily only when the analyzer is actually called.
    from esn.engine.analyzer import LLMAnalyzer

    analyzer = esn.make_agent_analyzer(model="claude-haiku-4-5-20251001")
    assert isinstance(analyzer, LLMAnalyzer)


def test_make_agent_predictor_returns_real_predictor():
    from esn.engine.predictor import LLMPredictor

    predictor = esn.make_agent_predictor(model="claude-haiku-4-5-20251001")
    assert isinstance(predictor, LLMPredictor)


def test_agent_client_raises_clear_error_without_agent_extra(monkeypatch):
    """The subscription client surfaces a clear ``[agent]``-extra error when
    ``claude_agent_sdk`` cannot be imported."""
    import esn.api as api

    # Patch the import machinery the client uses (``from claude_agent_sdk import ...``
    # resolves through builtins.__import__).
    import builtins

    real_builtins_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise ImportError("No module named 'claude_agent_sdk'")
        return real_builtins_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    client = api._AgentLLMClient(model="claude-haiku-4-5-20251001")
    with pytest.raises(RuntimeError, match=r"'\[agent\]' extra"):
        client("system", "user")


# ---------------------------------------------------------------------------
# Engine boundary: domain.evaluator must return an EvaluationResult
# ---------------------------------------------------------------------------


def test_evaluator_wrong_return_type_raises_clear_error():
    """A newcomer evaluator returning a bare int fails at the boundary with a
    message naming the contract, not deep with ``'int' has no attribute success``."""

    def _bad_evaluator(_artifact):
        return 42  # not an EvaluationResult

    domain = DomainSpec(
        name="bad-eval-toy",
        description="evaluator returns the wrong type",
        initial_code=SEED_CODE,
        compiler=_AlwaysCompileCompiler(),
        evaluator=_bad_evaluator,
    )
    engine = ESNEngine(domain=domain)

    with pytest.raises(ValueError, match="domain.evaluator must return an esn.EvaluationResult"):
        engine._evaluate_seed_if_needed()


# ---------------------------------------------------------------------------
# DomainSpec.__post_init__: program_interface vs initial_code
# ---------------------------------------------------------------------------


def test_solve_interface_requires_solve_function():
    with pytest.raises(ValueError, match="program_interface"):
        DomainSpec(
            name="missing-solve",
            description="solve interface but no solve() in initial_code",
            initial_code="def main():\n    pass\n",
            compiler=_AlwaysCompileCompiler(),
            evaluator=lambda a: a,
            program_interface="solve",
        )


def test_solve_interface_accepts_solve_function():
    # Must NOT raise — the real examples rely on this.
    DomainSpec(
        name="ok-solve",
        description="solve interface with a solve()",
        initial_code=SEED_CODE,
        compiler=_AlwaysCompileCompiler(),
        evaluator=lambda a: a,
        program_interface="solve",
    )


def test_stdio_interface_rejects_solve_function():
    with pytest.raises(ValueError, match="program_interface"):
        DomainSpec(
            name="stdio-with-solve",
            description="stdio interface but initial_code defines solve()",
            initial_code="def solve():\n    return 0\n",
            compiler=_AlwaysCompileCompiler(),
            evaluator=lambda a: a,
            program_interface="stdio",
        )
