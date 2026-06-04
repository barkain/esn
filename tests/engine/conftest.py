"""Shared fixtures for ESN engine tests."""

from __future__ import annotations

import pytest

from esn.core.models import EvaluationResult
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine


SIMPLE_INITIAL_CODE = """\
def solve():
    return [1, 2, 3]
"""


def _sum_evaluator(artifact):
    """Simple evaluator that sums a list artifact."""
    if artifact is None:
        return EvaluationResult(score=0.0, success=False)
    try:
        total = float(sum(artifact))
        return EvaluationResult(score=total, success=True)
    except Exception:
        return EvaluationResult(score=0.0, success=False)


@pytest.fixture()
def simple_compiler() -> PythonSandboxCompiler:
    """PythonSandboxCompiler with math allowed."""
    return PythonSandboxCompiler(allowed_imports=frozenset({"math"}))


@pytest.fixture()
def simple_evaluator():
    """Callable that sums a list artifact."""
    return _sum_evaluator


@pytest.fixture()
def simple_domain(simple_compiler, simple_evaluator) -> DomainSpec:
    """DomainSpec with initial_code that defines solve() returning [1,2,3]."""
    return DomainSpec(
        name="test",
        description="simple test domain",
        initial_code=SIMPLE_INITIAL_CODE,
        compiler=simple_compiler,
        evaluator=simple_evaluator,
        allowed_imports=frozenset({"math"}),
    )


@pytest.fixture()
def engine_no_llm(simple_domain) -> ESNEngine:
    """ESNEngine with no mutator/predictor/analyzer (tests pure engine logic)."""
    return ESNEngine(domain=simple_domain)
