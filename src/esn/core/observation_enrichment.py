# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Phase 3.9 — multi-aspect observation text enrichment.

Pure helper module: takes the data that is already available when a new
hypothesis is admitted to the knowledge bank, and returns a short
human-readable "aspects" tag string that is appended to the LLM-produced
hypothesis text BEFORE embedding.

The goal is not to replace the LLM-generated observation text — it is to
give the embedding model additional, deterministic discriminative signal
(family, operator, optimizer calls, code motifs, solve signature, failure
mode) so that the spectral bank separates strategies more cleanly.

Keep this module pure and dependency-free so it is trivial to unit-test.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Optimizer / search call patterns we look for in the candidate source.
# Kept explicit rather than clever — this is a discriminator, not a parser.
_OPTIMIZER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("basinhopping", "basinhopping"),
    ("differential_evolution", "diff-evolution"),
    ("dual_annealing", "dual-annealing"),
    ("shgo", "shgo"),
    ("least_squares", "least-squares"),
    ("scipy.optimize.minimize", "scipy.minimize"),
    ("minimize(", "scipy.minimize"),
    ("cma.CMAEvolutionStrategy", "cma-es"),
    ("simulated_annealing", "sim-anneal"),
    ("linprog", "linprog"),
    ("milp", "milp"),
)

# Structural code motifs — cheap keyword presence tests.
#
# The goal is discrimination between *strategies*, not exhaustive coverage.
# Patterns are grouped by role so future additions land in the right bucket.
_MOTIF_PATTERNS: tuple[str, ...] = (
    # --- numpy / scipy structural ---
    "np.linalg",
    "np.fft",
    "np.einsum",
    "np.meshgrid",
    "np.linspace",
    "itertools",
    "heapq",
    "networkx",
    "numba",
    "scipy.spatial",
    "KDTree",
    "cKDTree",
    "convex_hull",
    "Delaunay",
    "Voronoi",
    # --- geometry constants that fingerprint the layout ---
    "np.sqrt(3)",  # hex tiling
    "2*np.pi",  # ring / radial
    "2 * np.pi",  # ring / radial (spaced form)
    "137.5",  # phyllotaxis golden angle (deg)
    "2.399",  # phyllotaxis golden angle (rad)
    "golden",  # phyllotaxis
    "phyllotaxis",
    # --- refinement strategy keywords ---
    "gradient",
    "force",
    "repulsion",
    "repel",
    "anneal",
    "temperature",
    "perturb",
    "jitter",
    "nudge",
    "Lloyd",
    # --- control flow ---
    "while ",
)

# Cap the enrichment size so it cannot dominate the base hypothesis text
# in the embedding's token budget.
_MAX_ENRICHMENT_CHARS = 280

# Hardcoded-dump detection: post-polish artifacts are long tables of float
# literals with no loops or optimizer calls. Without a sentinel they look
# identical to each other regardless of the parent strategy, which collapses
# branch identity. Threshold is intentionally conservative.
_FLOAT_LITERAL_RE = re.compile(r"\b\d+\.\d+\b")
_LOOP_RE = re.compile(r"\bfor\s+\w+\s+in\b|\bwhile\s")
_DUMP_FLOAT_THRESHOLD = 20


def _detect_optimizers(code: str) -> list[str]:
    seen: list[str] = []
    for needle, label in _OPTIMIZER_PATTERNS:
        if needle in code and label not in seen:
            seen.append(label)
    return seen


def _looks_like_hardcoded_dump(code: str) -> bool:
    """True if `code` looks like a polished coordinate dump.

    Post-polish artifacts are long tables of float literals wrapped in a
    parameterless solve(). They have no loops and no optimizer calls.
    Used as a sentinel so raw LLM strategies and their polished siblings
    do not collapse to the same motif set.
    """
    if not code:
        return False
    if len(_FLOAT_LITERAL_RE.findall(code)) < _DUMP_FLOAT_THRESHOLD:
        return False
    if _LOOP_RE.search(code):
        return False
    for needle, _label in _OPTIMIZER_PATTERNS:
        if needle in code:
            return False
    return True


def _detect_motifs(code: str) -> list[str]:
    # Dump sentinel short-circuits other patterns. A coordinate dump may
    # still contain library imports (e.g. `import numpy as np`) but those
    # do not describe a *strategy*, so we hide them behind a single tag
    # that is distinct from every algorithmic motif.
    if _looks_like_hardcoded_dump(code):
        return ["hardcoded_dump"]
    return [m for m in _MOTIF_PATTERNS if m in code]


_SOLVE_SIG_RE = re.compile(r"def\s+solve\s*\(([^)]*)\)")


def _solve_signature(code: str) -> str | None:
    m = _SOLVE_SIG_RE.search(code)
    if not m:
        return None
    # Returns "" for parameterless solve() — still distinguishes code that
    # has a solve entry point from code that doesn't. The caller formats it.
    return m.group(1).strip()[:80]


def _summarize_failure(diagnostics: Any, errors: Iterable[str] | None) -> str | None:
    """Best-effort short failure tag."""
    if errors:
        first = next(iter(errors), None)
        if first:
            return str(first)[:80]
    if isinstance(diagnostics, dict):
        for key in ("error", "exception", "failure", "reason"):
            if diagnostics.get(key):
                return f"{key}={str(diagnostics[key])[:60]}"
        for key in ("overlap_count", "violations", "constraint_violations"):
            val = diagnostics.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return f"{key}={val}"
    return None


def build_observation_enrichment(
    *,
    code: str,
    score: float,
    style: str | None = None,
    intended_effect: str | None = None,
    family: str | None = None,
    success: bool = True,
    diagnostics: Any = None,
    errors: Iterable[str] | None = None,
) -> str:
    """Build a compact multi-aspect tag string for a candidate observation.

    All arguments are keyword-only so the call sites document themselves.
    Returns an empty string when no aspects are available — callers should
    treat empty as "do not enrich".
    """
    aspects: list[str] = []

    if family:
        aspects.append(f"family={family}")
    if style:
        aspects.append(f"operator={style}")
    if intended_effect:
        aspects.append(f"intent={intended_effect.strip()[:80]}")

    optimizers = _detect_optimizers(code or "")
    if optimizers:
        aspects.append("optimizer=" + ",".join(optimizers[:3]))

    motifs = _detect_motifs(code or "")
    if motifs:
        aspects.append("motifs=" + ",".join(motifs[:4]))

    sig = _solve_signature(code or "")
    if sig:
        aspects.append(f"signature=solve({sig})")

    aspects.append(f"score={score:.4f}")
    aspects.append("outcome=" + ("ok" if success else "fail"))

    fail_tag = _summarize_failure(diagnostics, errors)
    if fail_tag:
        aspects.append(f"failure={fail_tag}")

    if not aspects:
        return ""
    joined = " | ".join(aspects)
    if len(joined) > _MAX_ENRICHMENT_CHARS:
        joined = joined[: _MAX_ENRICHMENT_CHARS - 1] + "…"
    return joined


def enrich_hypothesis_text(base_text: str, enrichment: str) -> str:
    """Append an enrichment tag to a hypothesis text, additively.

    The original text is preserved as the first line so downstream readers
    (humans, the analyzer LLM) can still see the unmodified observation.
    """
    base = (base_text or "").rstrip()
    if not enrichment:
        return base
    return f"{base}\n[aspects: {enrichment}]"
