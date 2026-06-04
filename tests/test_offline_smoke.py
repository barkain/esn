# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Offline wiring smoke: esn.run mutate->compile->evaluate completes with no key.

Keeps the MockMutator path exercised in CI (the mock is TESTING-ONLY; user-facing
docs/examples demonstrate the real key-free agentic path instead). Uses the
in-process PythonSandboxCompiler so the loop needs no uv/network and no analyzer;
the no-analyzer RuntimeWarning is expected and suppressed here.
"""

from __future__ import annotations

import warnings
from typing import Any

import esn
from esn import (
    DomainSpec,
    EvaluationDiagnostics,
    EvaluationResult,
    MockMutator,
    PythonSandboxCompiler,
)

SEED_CODE = "def solve():\n    return [1, 2, 3]\n"


def _evaluate(artifact: Any) -> EvaluationResult:
    diagnostics = EvaluationDiagnostics()
    try:
        score = float(len(list(artifact)))
    except TypeError:
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)
    return EvaluationResult(score=score, success=True, diagnostics=diagnostics)


def _domain() -> DomainSpec:
    return DomainSpec(
        name="offline_smoke",
        description="Trivial list-length objective for an offline wiring smoke test.",
        initial_code=SEED_CODE,
        compiler=PythonSandboxCompiler(max_lines=200, timeout_seconds=10, seed=42),
        evaluator=_evaluate,
    )


def test_offline_run_completes_with_mock_mutator():
    with warnings.catch_warnings():
        # No analyzer => the loud INACTIVE-novelty warning is expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = esn.run(_domain(), generations=2, batch_size=2, mutator=MockMutator())

    assert result.generations == 2
    assert isinstance(result.best_score, float)
    assert isinstance(result.best_code, str) and "solve" in result.best_code
