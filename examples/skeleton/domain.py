# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Copy-me ESN domain template: a tiny, self-contained 0/1 knapsack.

This is the canonical starting point for onboarding a NEW task. Everything the
engine needs lives in this one file (no cross-file imports, no numpy, no `uv`
sandbox): inline instance data, an inline seed `solve()`, an inline evaluator,
and the `DomainSpec` that ties them together.

To make it your own, edit the `# ADAPT:` blocks and leave the `# KEEP:` blocks
(those are the ESN contract). Run it with: `uv run python examples/skeleton/run.py`.
"""

from __future__ import annotations

from typing import Any

from esn import (  # KEEP: import the public surface from esn.* only
    DomainSpec,
    EvaluationDiagnostics,
    EvaluationResult,
    PythonSandboxCompiler,
)

# ADAPT: your problem's instance data. Here: 10 items as (weight, value), and a
# capacity. The task is to pick a subset of items maximizing total value while
# total weight stays within CAPACITY.
ITEMS = [(2, 3), (3, 4), (4, 8), (5, 8), (9, 10), (7, 6), (1, 1), (6, 7), (8, 9), (10, 12)]
CAPACITY = 20

# ADAPT: the seed program. It must define `solve()` (the "solve" interface) and
# return the artifact your evaluator scores — here, a list of chosen item
# indices. This greedy value-density baseline is the score the LLM tries to beat.
SEED_CODE = """\
ITEMS = [(2, 3), (3, 4), (4, 8), (5, 8), (9, 10), (7, 6), (1, 1), (6, 7), (8, 9), (10, 12)]
CAPACITY = 20


def solve():
    chosen = []
    used = 0
    order = sorted(range(len(ITEMS)), key=lambda i: -ITEMS[i][1] / ITEMS[i][0])
    for i in order:
        weight = ITEMS[i][0]
        if used + weight <= CAPACITY:
            chosen.append(i)
            used += weight
    return chosen
"""


def evaluate(artifact: Any) -> EvaluationResult:
    # KEEP: an evaluator takes solve()'s return value and returns an
    # EvaluationResult. score is higher-is-better; success=False (e.g. on a
    # constraint violation) means the candidate is recorded but never promoted
    # to best. Be defensive: candidates may return malformed artifacts.
    diagnostics = EvaluationDiagnostics()
    try:
        # ADAPT: validate + score the artifact for YOUR problem below.
        indices = [int(i) for i in artifact]
    except (TypeError, ValueError):
        diagnostics.violations.append("solve() must return a list of item indices")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(ITEMS) for i in indices):
        diagnostics.violations.append("indices must be unique and within range")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    weight = sum(ITEMS[i][0] for i in indices)
    value = sum(ITEMS[i][1] for i in indices)
    if weight > CAPACITY:
        diagnostics.violations.append(f"total weight {weight} exceeds capacity {CAPACITY}")
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    diagnostics.residuals["total_value"] = float(value)
    return EvaluationResult(score=float(value), success=True, diagnostics=diagnostics)


def create_domain_spec() -> DomainSpec:
    return DomainSpec(
        name="knapsack_skeleton",  # ADAPT
        description="Pick items to maximize total value without exceeding the weight capacity.",  # ADAPT
        initial_code=SEED_CODE,  # KEEP: wire the seed in
        # KEEP: PythonSandboxCompiler runs solve() in-process (no uv, no network).
        # allowed_imports gates what the candidate may import; "math" is plenty here.
        compiler=PythonSandboxCompiler(
            allowed_imports=frozenset({"math"}),  # ADAPT for your problem
            max_lines=200,
            timeout_seconds=10,
            seed=42,
        ),
        evaluator=evaluate,  # KEEP: wire the evaluator in
        allowed_imports=frozenset({"math"}),  # ADAPT: also steers the LLM prompt + AST guard
        hard_constraints=[  # ADAPT: plain-language rules the LLM must respect
            "solve() returns a list of distinct item indices in range.",
            "Total weight of chosen items must not exceed the capacity.",
        ],
        hints=[
            "The greedy value-density baseline is not optimal; small swaps can do better."
        ],  # ADAPT
    )
