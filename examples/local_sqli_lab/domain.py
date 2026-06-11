# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""DomainSpec for the local SQLi lab (authorized CWE-89 security-testing example).

This wires the pure candidate (``initial.py``), the deliberately vulnerable
in-process target (``target.py``), and the non-gameable scorer (``evaluator.py``)
into one :class:`esn.DomainSpec`. It is **self-contained and offline**: the
evaluator talks only to an in-process SQLite lab, so it runs in CI with no
network and no external authorized target.

Candidate code runs under :class:`esn.UvSandboxCompiler`, i.e. in an **isolated
``uv`` subprocess**, and returns its artifact as JSON. This process boundary is
load-bearing for the security premise: the secret is generated per process inside
the *evaluator*, and a subprocess candidate cannot import or read the evaluator's
memory to recover it. (The in-process :class:`esn.PythonSandboxCompiler` is a
best-effort sandbox that an adversarial candidate can escape — e.g. via traceback
frame globals — to read the evaluator's secret directly, which would defeat the
whole point; so this example deliberately uses the stronger process isolation.)
"""

from __future__ import annotations

from esn import DomainSpec, UvSandboxCompiler

from .evaluator import evaluate_local_sqli_artifact
from .initial import INITIAL_SOLUTION

#: Safe standard-library modules the candidate may import. All are pure compute
#: (charsets, products, encodings, ranking) — nothing that does I/O, time, or DB
#: access. Target interaction lives entirely in the evaluator.
SAFE_PAYLOAD_IMPORTS = frozenset(
    {
        "itertools",
        "string",
        "base64",
        "binascii",
        "math",
        "re",
        "collections",
        "functools",
        "operator",
    }
)

_MAX_CODE_LINES = 240


def create_local_sqli_lab_domain_spec(*, timeout_seconds: int = 30) -> DomainSpec:
    """Build the local SQLi lab :class:`esn.DomainSpec`.

    Args:
        timeout_seconds: per-candidate sandbox timeout. The candidate is pure
            payload construction, but it runs in an isolated ``uv`` subprocess, so
            allow for subprocess spin-up.
    """
    return DomainSpec(
        name="local_sqli_lab",
        description=(
            "Authorized local SQL-injection lab (CWE-89). solve() is a PURE Python "
            "payload-strategy generator — it returns a plan of probe 'attempts' and "
            "performs no I/O. The evaluator runs those attempts only against an "
            "in-process, deliberately vulnerable SQLite endpoint and scores how close "
            "they come to exfiltrating a hidden secret. Higher score = closer to a "
            "working exploit; full credit requires the evaluator to reconstruct the "
            "entire secret character-by-character from the candidate's blind-injection "
            "template."
        ),
        initial_code=INITIAL_SOLUTION,
        compiler=UvSandboxCompiler(
            allowed_imports=SAFE_PAYLOAD_IMPORTS,
            max_lines=_MAX_CODE_LINES,
            timeout_seconds=timeout_seconds,
            seed=42,
        ),
        evaluator=evaluate_local_sqli_artifact,
        allowed_imports=SAFE_PAYLOAD_IMPORTS,
        max_code_lines=_MAX_CODE_LINES,
        program_interface="solve",
        hard_constraints=[
            "AUTHORIZED LOCAL LAB ONLY. Never target external or third-party systems.",
            "solve() must be PURE: no network, filesystem, subprocess, environment, "
            "sleep, or database access — the evaluator performs all target I/O.",
            "solve() returns {'target': 'local_sqli_lab', 'attempts': [...]} with at "
            "most a few dozen attempts.",
            "Each attempt has exactly one of: 'payload' (a direct SQL string), "
            "'boolean_template' (with a {predicate} slot), or 'payload_template' "
            "(a blind-extraction template with {n} and {qprefix} slots).",
            "Do not use destructive SQL (DROP/DELETE/UPDATE/INSERT/ATTACH/PRAGMA/...) "
            "or nondeterministic SQL (random(), datetime('now'), ...). Such attempts "
            "are rejected.",
        ],
        examples=[INITIAL_SOLUTION],
        hints=[
            "The endpoint runs: SELECT id, username FROM users WHERE active = 1 AND "
            "username = '<your input>'. It interpolates your input straight into the "
            "SQL (the vulnerability).",
            "There is a secrets table with a single 'canary' column holding an opaque "
            "hidden value. Recovering it fully is the goal.",
            "Partial signals all earn partial credit: a bare quote breaks the query "
            "(error), a boolean_template that truly flips row counts, a case-folded "
            "delay_ms(...) that adds latency. Build toward blind extraction from there.",
            "A toy filter blocks the exact strings 'UNION SELECT' and 'DELAY_MS', but "
            "SQLite keywords are case-insensitive — lower/mixed case still executes.",
            "Full credit comes ONLY from a blind-extraction payload_template the "
            "evaluator can drive: e.g. compare substr((select canary from secrets),1,{n}) "
            "to the evaluator-supplied {qprefix}, or gate delay_ms on that comparison. "
            "The evaluator reconstructs the secret itself, so hard-coding or guessing it "
            "earns nothing.",
        ],
        preferred_solution_shape=(
            "Write a bounded, pure Python payload-strategy generator (named helpers, "
            "loops, conditionals, charset products, ranking, or recursive prefix "
            "probing) — not a static list of literals. Return "
            "{'target': 'local_sqli_lab', 'attempts': [...]}. Do not perform target "
            "I/O; the evaluator owns all interaction. Prefer blind-extraction "
            "templates that read 'select canary from secrets' over one-off payloads."
        ),
    )
