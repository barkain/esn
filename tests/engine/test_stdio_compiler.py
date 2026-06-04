"""Integration tests for StdioCompiler.

These tests spawn real ``uv run`` subprocesses and require ``uv`` to be
installed on the machine.  They are skipped automatically when ``uv`` is
not on the PATH.
"""

from __future__ import annotations

import shutil

import pytest

from esn.engine.stdio_compiler import StdioCompiler

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compiler(**kwargs) -> StdioCompiler:
    """Create a compiler with sensible test defaults."""
    defaults = {"timeout_seconds": 10}
    defaults.update(kwargs)
    return StdioCompiler(**defaults)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_simple_stdin_stdout():
    """Program that reads stdin and writes it back."""
    compiler = _make_compiler()
    code = """\
import sys
data = sys.stdin.read()
print(data, end="")
"""
    result = compiler.compile(code, stdin_data="hello world")
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact == "hello world"


def test_stdin_with_processing():
    """Program that reads numbers from stdin, sums them, writes result."""
    compiler = _make_compiler()
    code = """\
import sys
lines = sys.stdin.read().strip().split()
total = sum(int(x) for x in lines)
print(total)
"""
    result = compiler.compile(code, stdin_data="10 20 30")
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact.strip() == "60"


def test_with_numpy():
    """Program that reads numbers, does numpy math, writes result."""
    compiler = _make_compiler()
    code = """\
import sys
import numpy as np
data = list(map(float, sys.stdin.read().strip().split()))
arr = np.array(data)
print(f"{arr.mean():.2f}")
"""
    result = compiler.compile(code, stdin_data="1.0 2.0 3.0 4.0")
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact.strip() == "2.50"


def test_no_stdin():
    """Program that needs no input - just writes output."""
    compiler = _make_compiler()
    code = """\
print("42")
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact.strip() == "42"


def test_multiline_output():
    """Program that writes multiple lines to stdout."""
    compiler = _make_compiler()
    code = """\
for i in range(5):
    print(i)
"""
    result = compiler.compile(code)
    assert result.success, f"Expected success, got errors: {result.errors}"
    lines = result.artifact.strip().splitlines()
    assert lines == ["0", "1", "2", "3", "4"]


def test_name_main_preserved():
    """if __name__ == '__main__' blocks should NOT be stripped for stdio programs."""
    compiler = _make_compiler()
    code = """\
import sys

def solve_problem():
    data = sys.stdin.read().strip()
    return data.upper()

if __name__ == '__main__':
    print(solve_problem())
"""
    result = compiler.compile(code, stdin_data="hello")
    assert result.success, f"Expected success, got errors: {result.errors}"
    assert result.artifact.strip() == "HELLO"


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


def test_syntax_error():
    """Invalid Python code."""
    compiler = _make_compiler()
    code = """\
def main(
    print("oops")
"""
    result = compiler.compile(code)
    assert not result.success
    assert result.artifact is None
    assert any("yntax" in e for e in result.errors)


def test_runtime_error():
    """Program that crashes at runtime."""
    compiler = _make_compiler()
    code = """\
x = 1 / 0
"""
    result = compiler.compile(code)
    assert not result.success
    assert result.artifact is None
    assert any("Runtime error" in e or "ZeroDivision" in e for e in result.errors)


def test_timeout():
    """Infinite loop program should time out."""
    compiler = _make_compiler(timeout_seconds=3)
    code = """\
while True:
    pass
"""
    result = compiler.compile(code)
    assert not result.success
    assert result.artifact is None
    assert any("imeout" in e for e in result.errors)


def test_forbidden_name():
    """Programs using forbidden names (eval, exec) are rejected at validation."""
    compiler = _make_compiler()
    code = """\
result = eval("1 + 2")
print(result)
"""
    result = compiler.compile(code)
    assert not result.success
    assert any("Forbidden name" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_seed():
    """Same seed should produce identical random output."""
    compiler = _make_compiler()
    code = """\
import random
vals = [random.random() for _ in range(5)]
print(" ".join(f"{v:.6f}" for v in vals))
"""
    r1 = compiler.compile(code, seed=123)
    r2 = compiler.compile(code, seed=123)
    assert r1.success and r2.success
    assert r1.artifact == r2.artifact, (
        f"Same seed should produce identical results: {r1.artifact!r} != {r2.artifact!r}"
    )

    r3 = compiler.compile(code, seed=999)
    assert r3.success
    assert r3.artifact != r1.artifact, "Different seeds should produce different results"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_on_success():
    """Successful compilation should include expected metadata."""
    compiler = _make_compiler()
    code = 'print("ok")\n'
    result = compiler.compile(code, seed=7)
    assert result.success
    assert result.metadata["stage"] == "complete"
    assert result.metadata["seed"] == 7
    assert result.metadata["runner"] == "stdio_subprocess"
    assert "code_hash" in result.metadata
