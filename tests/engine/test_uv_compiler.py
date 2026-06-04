"""Integration tests for UvSandboxCompiler.

These tests spawn real ``uv run`` subprocesses and require ``uv`` to be
installed on the machine.  They are skipped automatically when ``uv`` is
not on the PATH.
"""

from __future__ import annotations

import shutil

import pytest

from esn.engine.uv_compiler import UvSandboxCompiler

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compiler(**kwargs) -> UvSandboxCompiler:
    """Create a compiler with sensible test defaults."""
    defaults = {"timeout_seconds": 10, "seed": 42}
    defaults.update(kwargs)
    return UvSandboxCompiler(**defaults)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_simple_program():
    compiler = _make_compiler()
    code = """
def solve():
    return [1, 2, 3]
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact == [1, 2, 3]


def test_numpy_program():
    # No need to declare allowed_imports — UvSandboxCompiler extracts them from code
    compiler = _make_compiler()
    code = """
import numpy as np
def solve():
    return np.array([1.0, 2.0, 3.0])
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    # numpy arrays are serialized to plain lists via JSON
    assert result.artifact == [1.0, 2.0, 3.0]
    assert isinstance(result.artifact, list)


def test_tuple_preservation():
    compiler = _make_compiler()
    code = """
def solve():
    return ([1, 2], [3, 4])
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert isinstance(result.artifact, tuple)
    assert result.artifact == ([1, 2], [3, 4])


# ---------------------------------------------------------------------------
# Validation failure tests
# ---------------------------------------------------------------------------


def test_any_import_allowed():
    """UvSandboxCompiler does NOT restrict imports — uv installs on the fly."""
    compiler = _make_compiler()
    code = """
import math
def solve():
    return math.pi
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert abs(result.artifact - 3.14159) < 0.001


def test_syntax_error():
    compiler = _make_compiler()
    code = """
def solve(
    return 42
"""
    result = compiler.compile(code)
    assert not result.success
    assert result.artifact is None
    assert any("yntax" in e for e in result.errors)


def test_missing_solve():
    """Code without solve() -- the compiler now properly propagates runner failures."""
    compiler = _make_compiler()
    code = """
def helper():
    return 42
"""
    result = compiler.compile(code)
    assert result.success is False
    assert result.artifact is None


# ---------------------------------------------------------------------------
# Runtime failure tests
# ---------------------------------------------------------------------------


def test_runtime_error():
    """solve() raises -- the compiler now properly propagates runner failures."""
    compiler = _make_compiler()
    code = """
def solve():
    raise ValueError("intentional boom")
"""
    result = compiler.compile(code)
    assert result.success is False
    assert result.artifact is None


def test_timeout():
    compiler = _make_compiler(timeout_seconds=3)
    code = """
def solve():
    while True:
        pass
"""
    result = compiler.compile(code)
    assert not result.success
    assert result.artifact is None
    assert any("imeout" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_seed():
    compiler = _make_compiler()
    code = """
import random
def solve():
    return [random.random() for _ in range(5)]
"""
    r1 = compiler.compile(code, seed=123)
    r2 = compiler.compile(code, seed=123)
    assert r1.success and r2.success
    assert r1.artifact == r2.artifact, (
        f"Same seed should produce identical results: {r1.artifact} != {r2.artifact}"
    )

    # Different seed should (almost certainly) produce different results
    r3 = compiler.compile(code, seed=999)
    assert r3.success
    assert r3.artifact != r1.artifact, "Different seeds should produce different results"


# ---------------------------------------------------------------------------
# Third-party dependency installation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scipy_import():
    compiler = _make_compiler(
        timeout_seconds=120,  # scipy download can be slow
    )
    code = """
import scipy
def solve():
    return scipy.__version__
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert isinstance(result.artifact, str)
    assert len(result.artifact) > 0
