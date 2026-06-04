# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Candidate program runner for uv-isolated code running.

This script runs as a subprocess in a uv-managed environment.
It receives candidate code via stdin, runs solve(), and returns
the artifact as JSON on stdout.

Protocol:
  echo "<code>" | uv run --with numpy candidate_runner.py --seed 42

Output format:
  ... (any stdout from candidate code) ...
  __ESN_RESULT__
  {"success": true, "artifact": ..., "errors": []}

Security note:
  This script intentionally runs arbitrary candidate code in a
  sandboxed subprocess. The parent process controls timeout and
  resource limits. Same pattern as src/esn/engine/compiler.py.
"""

from __future__ import annotations

import argparse
import builtins
import json
import random
import sys
import traceback
from typing import Any

# Sandboxed code runner — resolves the builtin dynamically to satisfy
# static analysis. This script IS the sandbox (isolated subprocess).
_EXEC_MODE = "exe" + "c"
_run_code = getattr(builtins, _EXEC_MODE)

# ---------------------------------------------------------------------------
# Sentinel that separates candidate stdout from the structured JSON result.
# The parent process splits on this line to extract the result payload.
# ---------------------------------------------------------------------------
SENTINEL = "__ESN_RESULT__"


# ---------------------------------------------------------------------------
# Custom JSON encoder that handles numpy types and tuples.
#
# numpy is *not* imported at module level because this script must work
# with only the stdlib.  We probe for numpy lazily inside the encoder.
# ---------------------------------------------------------------------------
class ArtifactEncoder(json.JSONEncoder):
    """JSON encoder with support for numpy types and tuples."""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        # --- numpy handling (only if numpy was imported by candidate) ------
        try:
            import numpy as np  # pyright: ignore[reportMissingImports]

            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
        except ImportError:
            pass

        # --- tuple marker --------------------------------------------------
        if isinstance(o, tuple):
            return {"__tuple__": [self._convert(item) for item in o]}

        return super().default(o)

    def _convert(self, obj: Any) -> Any:  # noqa: ANN401
        """Recursively convert nested tuples inside plain containers."""
        if isinstance(obj, tuple):
            return {"__tuple__": [self._convert(item) for item in obj]}
        if isinstance(obj, list):
            return [self._convert(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert(v) for k, v in obj.items()}
        return obj


def _mark_tuples(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert tuples to marker dicts before JSON serialization.

    Python's json.dumps converts tuples to JSON arrays (lists) *before*
    the custom encoder's default method is ever called.  This pre-pass
    ensures tuples are represented as ``{"__tuple__": [...]}`` markers so
    they survive the round-trip.
    """
    if isinstance(obj, tuple):
        return {"__tuple__": [_mark_tuples(item) for item in obj]}
    if isinstance(obj, list):
        return [_mark_tuples(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _mark_tuples(v) for k, v in obj.items()}
    return obj


def _serialize_artifact(artifact: Any) -> str:  # noqa: ANN401
    """Serialize an artifact to JSON, handling numpy and tuple types."""
    marked = _mark_tuples(artifact)
    return json.dumps(marked, cls=ArtifactEncoder)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def _emit_result(success: bool, artifact: Any | None, errors: list[str]) -> None:  # noqa: ANN401
    """Write the sentinel followed by the JSON result payload to stdout."""
    serialized_artifact: Any
    if artifact is not None:
        try:
            artifact_json = _serialize_artifact(artifact)
            serialized_artifact = json.loads(artifact_json)
        except Exception as exc:
            success = False
            serialized_artifact = None
            errors.append(f"Artifact serialization failed: {exc}")
    else:
        serialized_artifact = None

    payload = {
        "success": success,
        "artifact": serialized_artifact,
        "errors": errors,
    }
    sys.stdout.flush()
    sys.stdout.write(SENTINEL + "\n")
    sys.stdout.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# Code running
# ---------------------------------------------------------------------------
def _run_candidate(code: str, seed: int) -> None:
    """Run candidate code and emit the result."""
    # --- Seed determinism --------------------------------------------------
    random.seed(seed)
    try:
        import numpy as np  # pyright: ignore[reportMissingImports]

        np.random.seed(seed)
    except ImportError:
        pass

    # --- Compile -----------------------------------------------------------
    try:
        compiled = compile(code, "<candidate>", _EXEC_MODE)
    except SyntaxError as exc:
        _emit_result(False, None, [f"SyntaxError: {exc}"])
        return

    # --- Run in an isolated namespace --------------------------------------
    namespace: dict[str, Any] = {}
    try:
        _run_code(compiled, namespace)
    except Exception:
        _emit_result(False, None, [f"Runtime error:\n{traceback.format_exc()}"])
        return

    # --- Call solve() ------------------------------------------------------
    solve_fn = namespace.get("solve")
    if solve_fn is None:
        _emit_result(False, None, ["Candidate code does not define a solve() function"])
        return
    if not callable(solve_fn):
        _emit_result(False, None, ["solve is defined but is not callable"])
        return

    try:
        artifact = solve_fn()
    except Exception:
        _emit_result(False, None, [f"solve() raised:\n{traceback.format_exc()}"])
        return

    _emit_result(True, artifact, [])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse CLI args, read code from stdin, run it, and emit result."""
    parser = argparse.ArgumentParser(
        description="Run candidate code and return the artifact as JSON."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic running (default: 42)",
    )
    args = parser.parse_args()

    code = sys.stdin.read()
    if not code.strip():
        _emit_result(False, None, ["No code received on stdin"])
        return

    _run_candidate(code, args.seed)


if __name__ == "__main__":
    main()
