# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Subprocess-based compiler for stdin/stdout programs using uv isolation.

Designed for AHC/ALE-Bench tasks where candidates are standalone programs that
read problem input from stdin and write solutions to stdout.  No ``solve()``
convention, no JSON artifact serialization — stdout text IS the artifact.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import tempfile
from pathlib import Path

from esn.core.models import CompilerResult  # type: ignore[import-untyped]
from esn.engine.compiler import _FORBIDDEN_ATTRS, _FORBIDDEN_NAMES  # type: ignore[import-untyped]
from esn.engine.subprocess_limiter import subprocess_slot
from esn.engine.uv_compiler import _STDLIB_MODULES, _IMPORT_TO_PACKAGE, _extract_imports  # type: ignore[import-untyped]


def _validate_stdio_ast(code: str, max_lines: int | None = None) -> list[str]:
    """Validate program AST for stdin/stdout execution mode.

    Lighter than ``validate_program_ast``: no import restrictions, no
    ``solve()`` requirement, and ``if __name__`` blocks are preserved
    (they ARE the program for stdin/stdout candidates).

    Checks:
    - Syntax validity
    - Line count (optional)
    - Forbidden names (``exec``, ``eval``, ``__import__``, etc.)
    - Forbidden attributes (dunder escapes)
    """
    errors: list[str] = []

    if max_lines is not None:
        lines = code.strip().splitlines()
        if len(lines) > max_lines:
            errors.append(f"Program exceeds {max_lines} line limit ({len(lines)} lines)")
            return errors

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            errors.append(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            errors.append(f"Forbidden attribute: {node.attr}")

    return errors


def _pip_deps_for(imports: set[str]) -> list[str]:
    """Resolve import names to pip package names, skipping stdlib."""
    deps: list[str] = []
    for name in sorted(imports):
        if name in _STDLIB_MODULES:
            continue
        pkg = _IMPORT_TO_PACKAGE.get(name, name)
        deps.append(pkg)
    return deps


class StdioCompiler:
    """Run candidate programs with stdin/stdout interface in isolated uv environments.

    For AHC/ALE-Bench tasks where candidates are standalone programs that:
    - Read problem input from stdin
    - Write solution to stdout
    - No ``solve()`` function, no artifact serialization
    - The stdout text IS the answer

    The ``seed`` parameter sets ``PYTHONHASHSEED`` and seeds ``random`` /
    ``numpy`` via a preamble injected before the candidate code, so that
    programs using ``random`` are reproducible without requiring the
    candidate to handle seeding itself.
    """

    def __init__(
        self,
        timeout_seconds: int = 5,
        max_lines: int | None = None,
        python_version: str = "3.12",
    ) -> None:
        self._timeout = timeout_seconds
        self._max_lines = max_lines
        self._python_version = python_version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(self, code_path: Path, deps: list[str]) -> list[str]:
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
        cmd.extend(["--", "python", str(code_path)])
        return cmd

    @staticmethod
    def _seed_preamble(seed: int) -> str:
        """Return a short preamble that seeds random and numpy for reproducibility."""
        return (
            f"import random as __esn_random; __esn_random.seed({seed})\n"
            f"try:\n"
            f"    import numpy as __esn_np; __esn_np.random.seed({seed})\n"
            f"except ImportError:\n"
            f"    pass\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, code: str, stdin_data: str = "", seed: int = 42) -> CompilerResult:
        """Execute program with *stdin_data* piped in, return stdout as artifact.

        Args:
            code: Python source — a standalone program that reads stdin / writes stdout.
            stdin_data: Data to pipe into the program's stdin.
            seed: Random seed for reproducibility (injected as preamble).

        Returns:
            ``CompilerResult`` with ``artifact=stdout_text`` on success.
        """
        # Step 1: Validate AST
        errors = _validate_stdio_ast(code, self._max_lines)
        if errors:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=errors,
                metadata={"stage": "validation"},
            )

        # Step 2: Resolve pip dependencies from code's actual imports
        deps = _pip_deps_for(_extract_imports(code))

        # Step 3: Write code to a temp file with seed preamble
        full_code = self._seed_preamble(seed) + "\n" + code

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="esn_stdio_",
            delete=False,
        ) as f:
            f.write(full_code)
            code_path = Path(f.name)

        try:
            # Step 4: Build command and run
            cmd = self._build_command(code_path, deps)

            try:
                import os

                env = {**os.environ, "PYTHONHASHSEED": str(seed)}
                with subprocess_slot():
                    result = subprocess.run(  # noqa: S603 — controlled command
                        cmd,
                        input=stdin_data,
                        capture_output=True,
                        text=True,
                        timeout=self._timeout,
                        env=env,
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
                truncated = stderr_lines[-30:] if len(stderr_lines) > 30 else stderr_lines
                return CompilerResult(
                    artifact=None,
                    success=False,
                    errors=["Runtime error:\n" + "\n".join(truncated)],
                    metadata={"stage": "runtime_error"},
                )

            # Step 6: stdout IS the artifact
            stdout_text = result.stdout
            return CompilerResult(
                artifact=stdout_text,
                success=True,
                metadata={
                    "stage": "complete",
                    "seed": seed,
                    "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
                    "runner": "stdio_subprocess",
                },
            )
        finally:
            # Clean up temp file
            code_path.unlink(missing_ok=True)
