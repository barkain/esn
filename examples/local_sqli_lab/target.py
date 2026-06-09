# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Deliberately vulnerable in-process target for the local SQLi lab (CWE-89).

This is a self-contained, offline teaching/CI target. It is intentionally
vulnerable to SQL injection — it builds the query by string interpolation — but
it runs entirely in-process over in-memory SQLite: no network, no real sleep, no
filesystem. It is for ESN research and CI smoke tests only; it does not, and
cannot, reach any external system.

The evaluator pairs a *primary* and a *shadow* :class:`LocalSqliLab` with the
same schema and rows but different secret canaries. A genuine blind-extraction
oracle must behave differently on the two targets — the only thing that differs
between them is the secret — which is how the evaluator proves an injection is
real rather than gamed (see ``evaluator.py``). The canary value is supplied by
the evaluator and never appears in the prompt the mutator sees.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

#: Inputs longer than this are rejected outright (a crude size guard).
MAX_QUERY_LEN = 240


@dataclass
class TargetResponse:
    """The observable result of one request to the lab.

    Only these fields are observable to the evaluator; everything the scorer
    rewards is derived from them, never from candidate-supplied metadata.
    """

    status_code: int
    body: str = ""
    rows: list[tuple[int, str]] = field(default_factory=list)
    error: str = ""
    blocked: bool = False
    simulated_latency_ms: float = 5.0


class LocalSqliLab:
    """A deliberately vulnerable in-process CWE-89 endpoint over in-memory SQLite.

    ``search_user(q)`` interpolates ``q`` straight into a SQL string (the
    vulnerability). A toy *exact-case* blocklist creates a small WAF-bypass
    sub-problem: SQLite keywords are case-insensitive, so ``union select`` and
    ``delay_ms`` still execute even though ``UNION SELECT`` / ``DELAY_MS`` are
    blocked. ``delay_ms(n)`` is a registered SQLite function that adds ``n`` ms
    of *simulated* latency (it never really sleeps), giving a deterministic
    time-based side channel.
    """

    # Exact-case tokens the toy filter blocks. Case variation bypasses it.
    BLOCKED_EXACT = ("UNION SELECT", "DELAY_MS")
    # Genuinely dangerous tokens blocked in any case (defence in depth for the
    # lab itself — these are never needed to solve it).
    FORBIDDEN_ANY_CASE = (
        "drop",
        "attach",
        "pragma",
        "load_extension",
        "readfile",
        "writefile",
        "vacuum",
        # Nondeterministic SQL — defence in depth (the evaluator already rejects
        # these). A random/time-varying oracle would break the deterministic
        # primary/shadow differential the scorer relies on. Mirrors the
        # evaluator's NONDETERMINISTIC block.
        "random",
        "current_time",
        "current_date",
        "julianday",
        "strftime",
        "unixepoch",
        "'now'",
    )

    def __init__(self, canary: str) -> None:
        self._canary = canary

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (id INTEGER, username TEXT, active INTEGER)")
        conn.execute("CREATE TABLE secrets (id INTEGER, canary TEXT)")
        conn.executemany(
            "INSERT INTO users VALUES (?, ?, ?)",
            [(1, "alice", 1), (2, "bob", 1), (3, "admin", 0)],
        )
        conn.execute("INSERT INTO secrets VALUES (1, ?)", (self._canary,))
        return conn

    def search_user(self, q: str) -> TargetResponse:
        """Run one ``username`` lookup with ``q`` injected into the SQL string."""
        if not isinstance(q, str):
            return TargetResponse(400, "bad request", blocked=True)
        if len(q) > MAX_QUERY_LEN:
            return TargetResponse(413, "query too long", blocked=True)
        lowered = q.lower()
        if any(token in lowered for token in self.FORBIDDEN_ANY_CASE):
            return TargetResponse(400, "unsafe token blocked", blocked=True)
        if any(token in q for token in self.BLOCKED_EXACT):
            return TargetResponse(403, "toy filter blocked exact token", blocked=True)

        conn = self._connect()
        delays: list[float] = []

        def delay_ms(ms: float) -> int:
            # Accumulate *simulated* latency; never actually sleep (CI-safe).
            value = max(0.0, min(float(ms), 100.0))
            delays.append(value)
            return 1

        conn.create_function("delay_ms", 1, delay_ms)

        # The vulnerability: q is interpolated directly into the SQL string.
        sql = f"SELECT id, username FROM users WHERE active = 1 AND username = '{q}'"
        try:
            rows = [(int(r[0]), str(r[1])) for r in conn.execute(sql).fetchall()]
        except sqlite3.Error as exc:
            return TargetResponse(
                500,
                "sqlite error: " + str(exc),
                error=str(exc),
                simulated_latency_ms=5.0 + sum(delays),
            )
        finally:
            conn.close()

        body = "\n".join(f"{row_id}:{name}" for row_id, name in rows)
        return TargetResponse(
            200,
            body,
            rows=rows,
            simulated_latency_ms=5.0 + sum(delays),
        )
