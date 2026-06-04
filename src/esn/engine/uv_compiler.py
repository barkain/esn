# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Subprocess-based sandbox compiler using uv for isolated Python environments."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from esn.core.models import CompilerResult  # type: ignore[import-untyped]
from esn.engine.compiler import _strip_name_main_blocks, validate_program_ast  # type: ignore[import-untyped]
from esn.engine.subprocess_limiter import subprocess_slot

# Map import names to pip package names (only non-obvious mappings needed)
_IMPORT_TO_PACKAGE: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
}

# Stdlib modules that never need pip installation
_STDLIB_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "math",
        "itertools",
        "collections",
        "functools",
        "random",
        "heapq",
        "bisect",
        "operator",
        "decimal",
        "fractions",
        "statistics",
        "cmath",
        "copy",
        "json",
        "re",
        "string",
        "textwrap",
        "struct",
        "hashlib",
        "typing",
        "abc",
        "dataclasses",
        "enum",
        "pathlib",
        "os",
        "sys",
        "io",
        "time",
        "datetime",
        "array",
        "contextlib",
        "warnings",
        "threading",
        "concurrent",
        "queue",
        "socket",
        "signal",
    }
)


def _extract_imports(code: str) -> set[str]:
    """Parse *code* and return the set of top-level import module names.

    For ``import foo`` returns ``{"foo"}``.
    For ``from bar.baz import x`` returns ``{"bar"}``.
    Returns an empty set on parse failure (let downstream validation report it).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _deserialize_artifact(obj: Any) -> Any:
    """Recursively reverse tuple markers from JSON serialization.

    Converts ``{"__tuple__": [...]}`` back to ``tuple(...)`` and recurses
    into nested dicts and lists.
    """
    if isinstance(obj, dict):
        if "__tuple__" in obj and len(obj) == 1:
            return tuple(_deserialize_artifact(item) for item in obj["__tuple__"])
        return {k: _deserialize_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_artifact(item) for item in obj]
    return obj


class UvSandboxCompiler:
    """Run candidate code in an isolated uv-managed Python subprocess.

    Drop-in replacement for ``PythonSandboxCompiler``.  Instead of
    in-process code execution with signal-based timeout, this compiler
    spawns a fresh ``uv run`` process with only the declared dependencies,
    enforcing timeout via ``subprocess.run(timeout=...)``.

    Satisfies the ``ProgramCompiler`` protocol from ``esn.engine.protocols``.
    """

    def __init__(
        self,
        allowed_imports: frozenset[str] = frozenset(),
        max_lines: int | None = None,
        timeout_seconds: int = 30,
        seed: int = 42,
        cache_dir: Path | None = None,
        python_version: str = "3.12",
    ) -> None:
        self._allowed_imports = allowed_imports
        self._max_lines = max_lines
        self._timeout = timeout_seconds
        self._seed = seed
        self._cache_dir = cache_dir or Path.home() / ".cache" / "esn" / "uv-envs"
        self._python_version = python_version
        self._runner_path = Path(__file__).parent / "candidate_runner.py"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pip_deps_for(self, imports: set[str]) -> list[str]:
        """Resolve a set of import names to pip package names, skipping stdlib."""
        deps: list[str] = []
        for name in sorted(imports):
            if name in _STDLIB_MODULES:
                continue
            pkg = _IMPORT_TO_PACKAGE.get(name, name)
            deps.append(pkg)
        return deps

    def _build_command(self, deps: list[str], seed: int) -> list[str]:
        """Assemble the ``uv run`` command line."""
        cmd: list[str] = [
            "uv",
            "run",
            "--no-project",
            "--python",
            self._python_version,
        ]
        for dep in deps:
            cmd.extend(["--with", dep])
        cmd.extend(["--", "python", str(self._runner_path), "--seed", str(seed)])
        return cmd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, code: str, seed: int | None = None) -> CompilerResult:
        """Execute program code in an isolated uv subprocess.

        Args:
            code: Python source with a ``solve()`` function.
            seed: Override seed (default: use compiler's fixed seed).

        Returns:
            ``CompilerResult`` with the artifact or error details.
        """
        effective_seed = seed if seed is not None else self._seed

        # Step 0: Strip if __name__ == '__main__' blocks
        code = _strip_name_main_blocks(code)

        # Step 1: AST validation (unrestricted imports — uv installs on the fly)
        errors = validate_program_ast(code, self._max_lines, allowed_imports=None)
        if errors:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=errors,
                metadata={"stage": "validation"},
            )

        # Step 2: Resolve pip dependencies from code's actual imports
        deps = self._pip_deps_for(_extract_imports(code))

        # Step 3: Build command
        cmd = self._build_command(deps, effective_seed)

        # Step 4: Run subprocess (semaphore limits concurrency to cpu_count)
        try:
            with subprocess_slot():
                result = subprocess.run(  # noqa: S603 — controlled command
                    cmd,
                    input=code,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
        except subprocess.TimeoutExpired:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=[f"Timeout: execution exceeded {self._timeout}s"],
                metadata={"stage": "timeout"},
            )

        # Step 5: Check exit code
        if result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines()
            # Limit stderr to last 30 lines to keep error reports manageable
            truncated = stderr_lines[-30:] if len(stderr_lines) > 30 else stderr_lines
            return CompilerResult(
                artifact=None,
                success=False,
                errors=["Runtime error:\n" + "\n".join(truncated)],
                metadata={"stage": "runtime_error"},
            )

        # Step 6: Parse JSON result after sentinel
        stdout = result.stdout
        sentinel = "__ESN_RESULT__"
        idx = stdout.find(sentinel)
        if idx == -1:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=[
                    "Missing __ESN_RESULT__ sentinel in runner output",
                    f"stdout (last 500 chars): {stdout[-500:]}",
                ],
                metadata={"stage": "runtime_error"},
            )

        json_str = stdout[idx + len(sentinel) :].strip()
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as e:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=[f"JSON parse error: {e}", f"raw: {json_str[:300]}"],
                metadata={"stage": "runtime_error"},
            )

        # Step 6b: Check runner-level success flag
        if not payload.get("success", False):
            return CompilerResult(
                artifact=None,
                success=False,
                errors=payload.get("errors", ["Unknown runner error"]),
                metadata={"stage": "runtime_error"},
            )

        # Step 7: Deserialize artifact (reverse tuple markers)
        artifact = _deserialize_artifact(payload.get("artifact"))

        return CompilerResult(
            artifact=artifact,
            success=True,
            metadata={
                "stage": "complete",
                "seed": effective_seed,
                "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
                "runner": "uv_subprocess",
            },
        )
