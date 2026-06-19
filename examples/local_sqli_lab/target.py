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

#: SQLite progress-handler budget: the handler fires every ``_PROGRESS_OPS``
#: virtual-machine steps and aborts the statement once it has fired
#: ``_PROGRESS_BUDGET`` times. ~2M opcodes is far more than any legitimate lab
#: query needs but trips quickly on a runaway recursive CTE.
_PROGRESS_OPS = 1000
_PROGRESS_BUDGET = 2000

#: Max length of any string/blob a query may build (memory-DoS guard, 3.11+).
_MAX_SQLITE_BLOB = 1_000_000


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
        # Bulk allocators: cheap to write but can balloon memory in a few opcodes
        # (the progress handler bounds CPU, not allocation). Not needed for the lab.
        "zeroblob",
        "printf",
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
        # Bound the size of any string/blob the query can build (memory-DoS guard).
        # Connection.setlimit is Python 3.11+; on 3.10 the bulk-allocator token
        # block above is the backstop.
        if hasattr(conn, "setlimit"):
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_SQLITE_BLOB)
        delays: list[float] = []

        def delay_ms(ms: float) -> int:
            # Accumulate *simulated* latency; never actually sleep (CI-safe).
            value = max(0.0, min(float(ms), 100.0))
            delays.append(value)
            return 1

        conn.create_function("delay_ms", 1, delay_ms)

        # Opcode budget: abort statements that run too long so a pathological
        # payload (e.g. a recursive CTE) cannot burn evaluator CPU. The handler
        # fires every _PROGRESS_OPS virtual-machine steps; returning non-zero
        # interrupts the query (raising an sqlite3 error we report as a 500).
        steps = [0]

        def _progress() -> int:
            steps[0] += 1
            return 1 if steps[0] > _PROGRESS_BUDGET else 0

        conn.set_progress_handler(_progress, _PROGRESS_OPS)

        # The vulnerability: q is interpolated directly into the SQL string.
        sql = f"SELECT id, username FROM users WHERE active = 1 AND username = '{q}'"
        try:
            rows = [(int(r[0]), str(r[1])) for r in conn.execute(sql).fetchall()]
        except (sqlite3.Error, ValueError, TypeError) as exc:
            # A UNION-style injection can return rows whose shape/type differ from
            # (int id, str username) — e.g. a string in the id column. In a real
            # app that surfaces as a 500; treat it the same here so a clever payload
            # yields an error signal instead of crashing the evaluator.
            return TargetResponse(
                500,
                "sqlite error: " + str(exc),
                error=str(exc),
                simulated_latency_ms=5.0 + sum(delays),
            )
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()

        body = "\n".join(f"{row_id}:{name}" for row_id, name in rows)
        return TargetResponse(
            200,
            body,
            rows=rows,
            simulated_latency_ms=5.0 + sum(delays),
        )
