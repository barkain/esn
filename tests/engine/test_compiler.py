"""Tests for PythonSandboxCompiler (ESN engine)."""

from __future__ import annotations

from esn.engine.compiler import PythonSandboxCompiler


class TestValidCompilation:
    def test_valid_compile_returns_artifact(self, simple_compiler):
        code = "def solve():\n    return [1, 2, 3]\n"
        result = simple_compiler.compile(code)
        assert result.success is True
        assert result.artifact == [1, 2, 3]

    def test_compile_with_allowed_import(self, simple_compiler):
        code = "import math\ndef solve():\n    return math.pi\n"
        result = simple_compiler.compile(code)
        assert result.success is True
        assert abs(result.artifact - 3.14159265) < 0.001

    def test_restricted_builtins_available(self, simple_compiler):
        code = (
            "def solve():\n"
            "    xs = list(range(5))\n"
            "    return [len(xs), min(xs), max(xs), sorted(xs, reverse=True), list(enumerate(xs))]\n"
        )
        result = simple_compiler.compile(code)
        assert result.success is True
        assert result.artifact[0] == 5
        assert result.artifact[1] == 0
        assert result.artifact[2] == 4

    def test_deterministic_rng(self, simple_compiler):
        code = "def solve():\n    return [random.random() for _ in range(5)]\n"
        r1 = simple_compiler.compile(code, seed=42)
        r2 = simple_compiler.compile(code, seed=42)
        assert r1.success and r2.success
        assert r1.artifact == r2.artifact

    def test_deterministic_rng_different_seeds(self, simple_compiler):
        code = "def solve():\n    return [random.random() for _ in range(5)]\n"
        r1 = simple_compiler.compile(code, seed=42)
        r2 = simple_compiler.compile(code, seed=99)
        assert r1.success and r2.success
        assert r1.artifact != r2.artifact


class TestForbiddenConstructs:
    def test_forbidden_import_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "import os\ndef solve():\n    return 1\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("os" in e for e in result.errors)

    def test_forbidden_import_from_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "from pathlib import Path\ndef solve():\n    return 1\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("pathlib" in e for e in result.errors)

    def test_dunder_class_escape_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    return ().__class__.__bases__\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("__class__" in e or "__bases__" in e for e in result.errors)

    def test_dunder_subclasses_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    return int.__subclasses__()\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("__subclasses__" in e for e in result.errors)

    def test_getattr_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    return getattr([], 'append')\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("getattr" in e for e in result.errors)

    def test_eval_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    return eval('1+1')\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("eval" in e for e in result.errors)

    def test_exec_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    exec('pass')\n    return 1\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("exec" in e for e in result.errors)

    def test_async_function_rejected(self):
        compiler = PythonSandboxCompiler()
        code = "async def solve():\n    return 1\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("Async" in e or "async" in e for e in result.errors)

    def test_open_not_available(self):
        compiler = PythonSandboxCompiler()
        code = "def solve():\n    return open('file')\n"
        result = compiler.compile(code)
        assert result.success is False
        assert any("open" in e for e in result.errors)


class TestErrorHandling:
    def test_missing_solve_function(self, simple_compiler):
        code = "x = 42\n"
        result = simple_compiler.compile(code)
        assert result.success is False
        assert any("solve" in e for e in result.errors)

    def test_solve_not_callable(self, simple_compiler):
        code = "solve = 42\n"
        result = simple_compiler.compile(code)
        assert result.success is False
        assert any("callable" in e.lower() or "not callable" in e.lower() for e in result.errors)

    def test_syntax_error(self, simple_compiler):
        code = "def solve(\n"
        result = simple_compiler.compile(code)
        assert result.success is False
        assert any("Syntax" in e or "syntax" in e for e in result.errors)

    def test_size_limit(self):
        compiler = PythonSandboxCompiler(max_lines=200)
        lines = ["x = 1"] * 250 + ["def solve():\n    return x\n"]
        code = "\n".join(lines)
        result = compiler.compile(code)
        assert result.success is False
        assert any("line limit" in e or "limit" in e for e in result.errors)

    def test_runtime_error_captured(self, simple_compiler):
        code = "def solve():\n    raise ValueError('boom')\n"
        result = simple_compiler.compile(code)
        assert result.success is False
        assert any("ValueError" in e and "boom" in e for e in result.errors)
