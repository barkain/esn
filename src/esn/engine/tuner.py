# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Tuner: domain-agnostic, evaluator-guided refinement of a candidate.

A *tuner* takes a successful candidate and tries to raise its score using **only
the domain evaluator as an oracle** — no domain-specific knowledge. It is the
exploitation counterpart to ESN's novelty-driven exploration: novelty preserves
structurally-new candidates, and a tuner *matures* them so a genuinely better new
structure reveals its true potential instead of being discarded for poorly-tuned
constants.

The bundled :class:`ParameterTuner` is fully general: it treats the float
literals of any candidate program as a parameter vector and pattern-searches over
them against the evaluator. Because it uses nothing but the evaluator, it
generalizes to any domain ESN can express — unlike a hand-written domain solver,
which is task-specific and not admissible in a framework-vs-framework comparison.

Every evaluator call a tuner makes is reported via ``TuningResult.evals_used`` so
the cost can be counted against a shared search budget.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class TuningResult:
    """Outcome of a tuning attempt."""

    improved: bool  # whether a strictly-better variant was found
    code: str  # tuned program code (original if not improved)
    artifact: Any = None  # compiled artifact of the tuned code (None if not improved)
    score: float = 0.0  # score of the tuned code (original score if not improved)
    improvement_delta: float = 0.0  # tuned_score - original_score
    evals_used: int = 0  # evaluator calls spent (for budget accounting)
    method: str = ""  # short description of what was done


@runtime_checkable
class Tuner(Protocol):
    """Evaluator-guided, post-mutation candidate refiner (no LLM calls).

    Implementations must be domain-agnostic: they may call ``compile`` and
    ``evaluator`` but must not assume anything about what the candidate computes.
    """

    def tune(
        self,
        *,
        code: str,
        score: float,
        compile: Callable[[str], Any],
        evaluator: Callable[[Any], Any],
        seed: int = 42,
    ) -> TuningResult: ...


def _float_literals(tree: ast.AST) -> list[ast.Constant]:
    """Return the float-valued Constant nodes of *tree* in a stable order.

    Only floats are treated as tunable: integer literals are usually structural
    (counts, indices, ``range`` bounds) and perturbing them tends to break the
    program rather than refine it. ``ast.walk`` yields nodes in a deterministic
    order, so re-parsing the same source maps the parameter vector consistently.
    """
    out: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is float:
            out.append(node)
    return out


class ParameterTuner:
    """Domain-agnostic tuner: pattern-search a candidate's float literals.

    Extracts every float literal as a free parameter and runs a coordinate
    pattern search against the evaluator, keeping any strictly-better feasible
    variant. Self-contained (no scipy); the evaluator is the only oracle.

    Args:
        max_evals: hard cap on evaluator calls per candidate (budget control).
        steps: relative perturbation fractions, one per round (shrinking).
    """

    def __init__(
        self,
        max_evals: int = 16,
        steps: tuple[float, ...] = (0.1, 0.05, 0.02),
    ) -> None:
        self.max_evals = max_evals
        self.steps = steps

    def tune(
        self,
        *,
        code: str,
        score: float,
        compile: Callable[[str], Any],
        evaluator: Callable[[Any], Any],
        seed: int = 42,
    ) -> TuningResult:
        try:
            base = _float_literals(ast.parse(code))
        except SyntaxError:
            return TuningResult(improved=False, code=code, score=score)
        if not base:
            return TuningResult(improved=False, code=code, score=score, method="no float params")

        x0 = [n.value for n in base]
        n_params = len(x0)
        evals = 0

        def evaluate(vec: list[float]) -> tuple[float | None, str, Any]:
            nonlocal evals
            tree = ast.parse(code)
            nodes = _float_literals(tree)
            if len(nodes) != n_params:  # paranoia: order/count must match
                return None, code, None
            for node, val in zip(nodes, vec):
                node.value = float(val)
            new_code = ast.unparse(tree)
            cr = compile(new_code)
            if not getattr(cr, "success", False):
                return None, new_code, None
            er = evaluator(cr.artifact)
            evals += 1
            sc = er.score if getattr(er, "success", False) else None
            return sc, new_code, cr.artifact

        best_x = list(x0)
        best_score = score
        best_code: str | None = None  # cached so we never re-evaluate the winner
        best_artifact: Any = None
        for rnd in range(len(self.steps)):
            step = self.steps[rnd]
            improved_round = False
            for i in range(n_params):
                if evals >= self.max_evals:
                    break
                for sign in (1.0, -1.0):
                    if evals >= self.max_evals:
                        break
                    trial = list(best_x)
                    magnitude = abs(trial[i]) if trial[i] != 0.0 else 1.0
                    trial[i] = trial[i] + sign * step * magnitude
                    sc, cand_code, cand_artifact = evaluate(trial)
                    if sc is not None and sc > best_score + 1e-9:
                        best_score = sc
                        best_x = trial
                        best_code = cand_code
                        best_artifact = cand_artifact
                        improved_round = True
            if not improved_round or evals >= self.max_evals:
                break

        if best_code is not None and best_artifact is not None and best_score > score + 1e-9:
            return TuningResult(
                improved=True,
                code=best_code,
                artifact=best_artifact,
                score=best_score,
                improvement_delta=best_score - score,
                evals_used=evals,
                method=f"parameter pattern search ({n_params} floats, {evals} evals)",
            )
        return TuningResult(
            improved=False, code=code, score=score, evals_used=evals,
            method=f"no gain ({n_params} floats, {evals} evals)",
        )
