# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Deterministic sandbox compiler for ESN engine program execution."""

from __future__ import annotations

import ast
import hashlib
import random  # noqa: S311 — deterministic seeding for reproducibility, not crypto
import signal
from typing import Any

from esn.core.models import CompilerResult


# Safe builtins subset — no file I/O, no imports, no eval/exec
_SAFE_BUILTINS: dict[str, Any] = {
    # Types
    "True": True,
    "False": False,
    "None": None,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    # Built-in functions (safe subset)
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "chr": chr,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "format": format,
    "hash": hash,
    "hex": hex,
    "id": id,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,  # Allow print for debugging (captured by sandbox)
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "slice": slice,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
    # Exceptions (needed for try/except)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
}

# Names forbidden in AST (security)
_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "__builtins__",
        "exec",
        "eval",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "open",
        "breakpoint",
        "exit",
        "quit",
        "vars",
        "dir",
    }
)

_FORBIDDEN_ATTRS = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__qualname__",
        "__module__",
        "__dict__",
    }
)


def validate_program_ast(
    code: str,
    max_lines: int | None = None,
    allowed_imports: frozenset[str] | None = frozenset(),
) -> list[str]:
    """Validate program code via AST analysis.

    Args:
        code: Python source to validate.
        max_lines: Maximum allowed line count.  ``None`` (default) means
            **no limit** — the size check is skipped entirely.
        allowed_imports: Set of allowed top-level import names.
            ``frozenset()`` (default) means **no** imports are allowed.
            ``None`` means imports are **unrestricted** (skip import checks).

    Returns list of errors (empty = valid).
    """
    errors: list[str] = []

    # Size check (skip when no limit is set)
    if max_lines is not None:
        lines = code.strip().splitlines()
        if len(lines) > max_lines:
            errors.append(f"Program exceeds {max_lines} line limit ({len(lines)} lines)")
            return errors  # Don't bother parsing if too large

    # Parse check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return errors

    # AST walk for forbidden constructs
    for node in ast.walk(tree):
        # Forbidden names
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            errors.append(f"Forbidden name: {node.id}")
        # Forbidden attribute access (dunder escapes)
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            errors.append(f"Forbidden attribute: {node.attr}")
        # Import check — skipped when allowed_imports is None (unrestricted)
        if allowed_imports is not None:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed_imports:
                        errors.append(f"Disallowed import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in allowed_imports:
                    errors.append(f"Disallowed import: {node.module}")
        # Async functions not allowed
        if isinstance(node, ast.AsyncFunctionDef):
            errors.append(f"Async functions not allowed: {node.name}")

    return errors


def _strip_name_main_blocks(code: str) -> str:
    """Remove ``if __name__ == '__main__':`` blocks from source code.

    Defence-in-depth: even if an LLM emits such a block, we silently
    strip it so it cannot trigger NameError or execute test harness code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # Let downstream validation report the error

    new_body: list[ast.stmt] = []
    changed = False
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            changed = True
            continue
        new_body.append(node)

    if not changed:
        return code

    tree.body = new_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class PythonSandboxCompiler:
    """Executes program code deterministically in a restricted Python sandbox.

    Guarantees:
    - Deterministic: same code + same seed = same output
    - Sandboxed: restricted builtins, no file/network access
    - Bounded: timeout enforced via signal.alarm (Unix)
    - Auditable: captures errors, warnings, metadata
    """

    def __init__(
        self,
        allowed_imports: frozenset[str] = frozenset(),
        max_lines: int = 200,
        timeout_seconds: int = 30,
        seed: int = 42,
    ) -> None:
        self._allowed_imports = allowed_imports
        self._max_lines = max_lines
        self._timeout = timeout_seconds
        self._seed = seed

        # Pre-build the import namespace for allowed modules
        self._import_cache: dict[str, Any] = {}
        for mod_name in allowed_imports:
            try:
                self._import_cache[mod_name] = __import__(mod_name)
            except ImportError:
                pass  # Will fail at compile time if code uses it

    def compile(self, code: str, seed: int | None = None) -> CompilerResult:
        """Execute program code in sandbox and return the result.

        The program must define a `solve()` function that returns the artifact.
        The artifact is whatever the domain evaluator expects.

        Args:
            code: Python source code with a solve() function.
            seed: Override seed (default: use compiler's fixed seed).

        Returns:
            CompilerResult with artifact (solve() return value) or errors.
        """
        effective_seed = seed if seed is not None else self._seed

        # Step 0: Strip if __name__ == '__main__' blocks (defence-in-depth)
        code = _strip_name_main_blocks(code)

        # Step 1: Validate AST
        errors = validate_program_ast(code, self._max_lines, self._allowed_imports)
        if errors:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=errors,
                metadata={"stage": "validation"},
            )

        # Step 2: Build restricted namespace
        builtins = dict(_SAFE_BUILTINS)

        # Add a restricted __import__ that only allows whitelisted modules
        import_cache = self._import_cache

        def _restricted_import(name, *args, **kwargs):  # noqa: S102 — intentional sandbox import gate
            if name not in import_cache:
                raise ImportError(f"Import not allowed: {name}")
            return import_cache[name]

        builtins["__import__"] = _restricted_import
        namespace: dict[str, Any] = {"__builtins__": builtins, "__name__": "__main__"}

        # Add allowed imports (also directly in namespace for convenience)
        for mod_name, mod in self._import_cache.items():
            namespace[mod_name] = mod

        # Add deterministic RNG
        rng = random.Random(effective_seed)  # noqa: S311 — deterministic seed, not crypto
        namespace["random"] = rng  # Programs get a seeded Random instance, not the module

        # Step 3: Execute with timeout
        old_handler = None
        timed_out = False

        def _timeout_handler(signum, frame):
            nonlocal timed_out
            timed_out = True
            raise TimeoutError(f"Program execution exceeded {self._timeout}s timeout")

        try:
            # Set timeout (Unix only)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self._timeout)

            # Execute program definition
            exec(code, namespace)  # noqa: S102

            # Step 4: Call solve()
            if "solve" not in namespace:
                return CompilerResult(
                    artifact=None,
                    success=False,
                    errors=["Program must define a solve() function"],
                    metadata={"stage": "execution"},
                )

            if not callable(namespace["solve"]):
                return CompilerResult(
                    artifact=None,
                    success=False,
                    errors=["solve is not callable"],
                    metadata={"stage": "execution"},
                )

            artifact = namespace["solve"]()

            return CompilerResult(
                artifact=artifact,
                success=True,
                metadata={
                    "stage": "complete",
                    "seed": effective_seed,
                    "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
                },
            )

        except TimeoutError:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=[f"Timeout: execution exceeded {self._timeout}s"],
                metadata={"stage": "timeout"},
            )
        except Exception as e:
            return CompilerResult(
                artifact=None,
                success=False,
                errors=[f"Runtime error: {type(e).__name__}: {e}"],
                metadata={"stage": "runtime_error"},
            )
        finally:
            # Clean up timeout
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
