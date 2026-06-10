# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Non-gameable evaluator for the local SQLi lab (CWE-89).

The evaluator owns *all* target interaction and scoring. Candidate ``solve()``
output is treated as untrusted: every point on the ladder is derived from
**measured target behavior**, never from candidate-supplied metadata (``name``,
``kind``, ``expect``, ``guess`` …). This is what keeps the search honest — a
fitness search will exploit any oracle it can, so the oracle must not leak.

The continuous "distance-to-exploit" ladder (higher is better, 0..1000):

    0          infeasible / unsafe / out-of-scope            -> success=False
    25-75      valid in-scope plan, no signal yet
    150-190    a payload reaches the SQL parser (syntax 500)
    300-340    a SQLite error is fingerprinted
    520-640    boolean differential (evaluator-controlled predicate flip)
    560-620    timing primitive (a real case-folded delay_ms raises latency)
    625-980    blind canary recovered char-by-char (+25 per genuine char;
               crosses 700 only at ~4 genuinely-read characters)
    1000       full secret recovered (every character reconstructed by the evaluator)

The ``>= 700`` band is reserved *exclusively* for confirmed extraction of the
secret. "Means" capabilities (a yes/no oracle, a timing primitive) sit below it —
they are stepping stones, not exploits, and must never look like progress toward
the secret that they have not actually made.

Two anti-gaming devices make the blind/exfil rungs real:

* **Wrong-guess control.** A confirmed prefix must make the oracle positive for
  the candidate prefix *and* negative for a same-length wrong prefix — so credit
  tracks the actual secret, not a tautology.
* **Shadow target.** The same payload is run against a second lab whose only
  difference is a different canary; genuine extraction must *flip* between them.
  This kills "hardcode the answer / echo a literal" tricks for both blind
  extraction and full union disclosure.

The canary values live here, never in the prompt the mutator sees.
"""

from __future__ import annotations

import secrets
from typing import Any

from esn import EvaluationDiagnostics, EvaluationResult

from .target import LocalSqliLab


# Secrets used by the lab. They are generated FRESH per process from a CSPRNG,
# are never written into source or the prompt, and are unguessable opaque hex —
# so candidate code cannot know them and therefore cannot hard-code, split, or
# "CASE-WHEN" them. (A fixed source-known canary is the root of every disclosure
# cheat: whatever the value, a candidate could embed it in some encoding. Making
# it unknowable removes the entire class.) They stay fixed for the process
# lifetime, so scoring is deterministic within a single ESN run.
#   - PRIMARY_CANARY is the secret the search must recover;
#   - SHADOW_CANARY is a control on an otherwise-identical target, used to prove a
#     blind oracle genuinely depends on the secret (not a candidate constant).
# Full-exfil credit (1000) comes ONLY from the evaluator's own char-by-char
# reconstruction (_recover_prefix_from_template); there is no "canary appeared in
# a body" path. A candidate cannot fake an extraction the evaluator drives and
# verifies itself against a secret the candidate never sees.
def _new_canary() -> str:
    return secrets.token_hex(8)  # 16 opaque hex chars, unknowable to candidates


PRIMARY_CANARY = _new_canary()
SHADOW_CANARY = _new_canary()
while SHADOW_CANARY[0] == PRIMARY_CANARY[0]:  # distinct first char for a clean shadow control
    SHADOW_CANARY = _new_canary()
CANARIES = (PRIMARY_CANARY, SHADOW_CANARY)

# Characters the evaluator will try when reconstructing a blind prefix. A
# candidate-supplied charset is intersected with this set (order is a search hint
# only; it can never assert correctness).
SAFE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}_-$"

MAX_ATTEMPTS = 32
MAX_PAYLOAD_LEN = 240
MAX_TEMPLATE_DEPTH = 24
MAX_TEMPLATE_CALLS = 900  # total primary+wrong+shadow probes per candidate
TIMING_THRESHOLD_MS = 25.0

# Genuinely dangerous tokens; a candidate is rejected (success=False) if any
# attempt contains one. They are never needed to solve the lab.
DESTRUCTIVE = (
    "drop",
    "attach",
    "pragma",
    "load_extension",
    "readfile",
    "writefile",
    "vacuum",
    "delete",
    "insert",
    "update",
)

# Nondeterministic SQL the scorer must refuse: a random/time-varying oracle could
# pass the primary/wrong/shadow differential *by chance* and reach the extraction
# band without reading the secret. Reject these outright (success=False). The
# target mirrors this block as defence in depth.
NONDETERMINISTIC = (
    "random",  # random(), randomblob()
    "current_time",  # also matches current_timestamp
    "current_date",
    "julianday",
    "strftime",
    "unixepoch",
    "'now'",
)

_TRUE_PREDICATE = "1=1"
_FALSE_PREDICATE = "1=0"

# Substrings that mark a genuine SQLite engine error message (status 500). Used
# to fingerprint a real DB error rather than a string the candidate echoed.
_SQLITE_ERR_KW = ("syntax error", "unrecognized token", "no such", "incomplete", "near ")


def _contains_canary_literal(value: str) -> bool:
    # A candidate must *read* the secret, never embed it. (Canaries are opaque
    # and absent from the prompt, so any literal occurrence is a hard-coded
    # answer — reject it rather than award disclosure credit.)
    return any(canary in value for canary in CANARIES)


# ---------------------------------------------------------------------------
# Validation / coercion (the viability gate)
# ---------------------------------------------------------------------------


_SCHEMA_KEYS = ("payload", "boolean_template", "payload_template")


def _coerce_attempts(artifact: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the artifact and return (normalized attempts, errors).

    A non-empty ``errors`` list means the attempt is infeasible/unsafe and the
    evaluator returns ``success=False`` (never eligible to become best).
    """
    errors: list[str] = []
    raw_attempts: Any
    if isinstance(artifact, dict):
        if artifact.get("target") != "local_sqli_lab":
            errors.append("artifact 'target' must be 'local_sqli_lab'")
        raw_attempts = artifact.get("attempts")
    else:
        raw_attempts = artifact

    if not isinstance(raw_attempts, list) or not raw_attempts:
        errors.append("solve() must return a non-empty 'attempts' list")
        return [], errors
    if len(raw_attempts) > MAX_ATTEMPTS:
        errors.append(f"too many attempts: {len(raw_attempts)} > {MAX_ATTEMPTS}")
        return [], errors

    attempts: list[dict[str, Any]] = []
    for i, item in enumerate(raw_attempts):
        if isinstance(item, str):
            item = {"payload": item}
        if not isinstance(item, dict):
            errors.append(f"attempt {i} must be a dict or string")
            continue
        present = [k for k in _SCHEMA_KEYS if k in item]
        if len(present) != 1:
            errors.append(
                f"attempt {i} must have exactly one of payload/boolean_template/payload_template"
            )
            continue
        value = item[present[0]]
        if not isinstance(value, str) or not value:
            errors.append(f"attempt {i} '{present[0]}' must be a non-empty string")
            continue
        # Scan EVERY string field of the attempt, not just the SQL field: an
        # unsafe token or an embedded secret literal is a hard-coded answer
        # wherever it hides (e.g. in 'charset' or 'name'), so reject it there too.
        bad = None
        for key, field in item.items():
            if not isinstance(field, str):
                continue
            if len(field) > MAX_PAYLOAD_LEN:
                bad = f"attempt {i} field '{key}' too long ({len(field)} > {MAX_PAYLOAD_LEN})"
            elif any(token in field.lower() for token in DESTRUCTIVE):
                bad = f"attempt {i} field '{key}' contains a blocked destructive token"
            elif any(token in field.lower() for token in NONDETERMINISTIC):
                bad = f"attempt {i} field '{key}' uses a nondeterministic SQL function"
            elif _contains_canary_literal(field):
                bad = f"attempt {i} field '{key}' embeds a secret literal (read it, don't hard-code it)"
            if bad:
                break
        if bad:
            errors.append(bad)
            continue
        attempts.append(item)
    if not attempts and not errors:
        errors.append("no valid attempts")
    return attempts, errors


# ---------------------------------------------------------------------------
# Oracles (behavior-derived; metadata never participates)
# ---------------------------------------------------------------------------


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _render_template(template: str, prefix: str) -> str:
    if "{qprefix}" not in template or "{n}" not in template:
        raise ValueError("blind template must contain {n} and {qprefix}")
    return template.replace("{n}", str(len(prefix))).replace("{qprefix}", _sql_literal(prefix))


def _wrong_same_length(prefix: str) -> str:
    if not prefix:
        return "X"
    last = prefix[-1]
    replacement = "X" if last != "X" else "Y"
    return prefix[:-1] + replacement


def _oracle_channels(response, baseline_rows: int, baseline_latency_ms: float) -> set[str]:
    """The observable positive channels for one response (row count, timing)."""
    channels: set[str] = set()
    if response.status_code == 200 and len(response.rows) > baseline_rows:
        channels.add("row")
    if (
        response.status_code == 200
        and response.simulated_latency_ms - baseline_latency_ms >= TIMING_THRESHOLD_MS
    ):
        channels.add("timing")
    return channels


def _confirmed_channels_for_prefix(
    *,
    template: str,
    prefix: str,
    primary: LocalSqliLab,
    shadow: LocalSqliLab,
    baseline_rows: int,
    baseline_latency_ms: float,
) -> set[str]:
    """Channels that prove ``prefix`` matches the *primary* secret.

    Positive on primary+prefix, but negative for both a wrong same-length prefix
    (so it depends on the guess) and the shadow target (so it depends on the
    actual secret, not a candidate-controlled constant). Tautologies and
    hardcoded-literal tricks net out to the empty set.
    """
    payload = _render_template(template, prefix)
    wrong = _render_template(template, _wrong_same_length(prefix))

    primary_ch = _oracle_channels(primary.search_user(payload), baseline_rows, baseline_latency_ms)
    wrong_ch = _oracle_channels(primary.search_user(wrong), baseline_rows, baseline_latency_ms)
    shadow_ch = _oracle_channels(shadow.search_user(payload), baseline_rows, baseline_latency_ms)
    return primary_ch - wrong_ch - shadow_ch


def _recover_prefix_from_template(
    *,
    template: str,
    charset: str,
    max_depth: int,
    primary: LocalSqliLab,
    shadow: LocalSqliLab,
    baseline_rows: int,
    baseline_latency_ms: float,
    call_budget: int,
) -> tuple[str, set[str], int]:
    """Reconstruct the longest prefix the *evaluator* can confirm via the template."""
    ordered = [ch for ch in charset if ch in SAFE_CHARSET]
    # Append any remaining safe chars so a thin candidate charset cannot cap us.
    seen = set(ordered)
    ordered.extend(ch for ch in SAFE_CHARSET if ch not in seen)
    max_depth = max(1, min(max_depth, MAX_TEMPLATE_DEPTH))

    recovered = ""
    channels_seen: set[str] = set()
    calls = 0
    for _pos in range(max_depth):
        found: tuple[str, set[str]] | None = None
        for ch in ordered:
            if calls + 3 > call_budget:
                return recovered, channels_seen, calls
            channels = _confirmed_channels_for_prefix(
                template=template,
                prefix=recovered + ch,
                primary=primary,
                shadow=shadow,
                baseline_rows=baseline_rows,
                baseline_latency_ms=baseline_latency_ms,
            )
            calls += 3
            if channels:
                found = (ch, channels)
                break
        if found is None:
            break
        recovered += found[0]
        channels_seen |= found[1]
    return recovered, channels_seen, calls


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _boolean_flips(template: str, primary: LocalSqliLab, baseline_rows: int) -> bool:
    """Credit only a measured row-count flip under evaluator-controlled predicates."""
    try:
        true_payload = template.replace("{predicate}", _TRUE_PREDICATE)
        false_payload = template.replace("{predicate}", _FALSE_PREDICATE)
    except Exception:  # noqa: BLE001
        return False
    if "{predicate}" not in template:
        return False
    true_rows = len(primary.search_user(true_payload).rows)
    false_rows = len(primary.search_user(false_payload).rows)
    return true_rows > baseline_rows and false_rows <= baseline_rows


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def evaluate_local_sqli_artifact(artifact: Any) -> EvaluationResult:
    diagnostics = EvaluationDiagnostics(
        constraints={
            "cwe": "CWE-89",
            "target": "local_in_process_sqlite",
            "max_attempts": float(MAX_ATTEMPTS),
            "max_payload_len": float(MAX_PAYLOAD_LEN),
        }
    )

    attempts, errors = _coerce_attempts(artifact)
    if errors:
        diagnostics.violations.extend(errors[:5])
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    # Primary target holds the secret to recover; the shadow target is identical
    # except for its canary and is used as a control to prove a blind oracle truly
    # depends on the secret.
    primary = LocalSqliLab(PRIMARY_CANARY)
    shadow = LocalSqliLab(SHADOW_CANARY)
    baseline = primary.search_user("__no_such_user__")
    baseline_rows = len(baseline.rows)
    baseline_latency_ms = baseline.simulated_latency_ms

    direct = [(a, str(a["payload"])) for a in attempts if isinstance(a.get("payload"), str)]
    bool_templates = [
        str(a["boolean_template"]) for a in attempts if isinstance(a.get("boolean_template"), str)
    ]
    blind_templates = [a for a in attempts if isinstance(a.get("payload_template"), str)]

    score = 25.0 + min(50.0, 2.5 * len(attempts))
    rung = "valid_plan"

    def bump(candidate_score: float, name: str) -> None:
        # Raise the score and re-label the rung only when this capability is the
        # new best — so 'rung' always names the *highest-scoring* signal, never a
        # lower one that happened to be evaluated later.
        nonlocal score, rung
        if candidate_score > score:
            score = candidate_score
            rung = name

    # --- Direct-probe rungs (all behavior-derived) ---
    direct_responses = [(a, payload, primary.search_user(payload)) for a, payload in direct]

    n_syntax = sum(1 for _a, _p, r in direct_responses if r.status_code == 500)
    if n_syntax:
        bump(150.0 + min(40.0, 10.0 * n_syntax), "quote_reaches_sql")

    # A real SQLite error (status 500 with an engine error message), not a string
    # the candidate echoed into the body.
    n_dberr = sum(
        1
        for _a, _p, r in direct_responses
        if r.status_code == 500 and any(kw in r.error.lower() for kw in _SQLITE_ERR_KW)
    )
    if n_dberr:
        bump(300.0 + min(40.0, 10.0 * n_dberr), "db_error_fingerprint")

    # Boolean differential — evaluator-controlled predicate flip only.
    n_flips = sum(1 for t in bool_templates if _boolean_flips(t, primary, baseline_rows))
    if n_flips:
        bump(520.0 + min(120.0, 40.0 * (n_flips - 1)), "boolean_differential")

    # Timing primitive: a real case-folded delay_ms injection raises *measured*
    # latency. Genuine and non-gameable — only an executed lowercase delay_ms can
    # move the clock past the DELAY_MS filter — but it is only a *means*, so it
    # scores BELOW the >=700 extraction band (it has not read the secret).
    # (A toy-filter-bypass rung was deliberately removed: token-presence credit
    # is gameable with a string literal, and a bypass that extracts nothing makes
    # no progress toward the secret. The bypass capability is credited only
    # through the rungs that actually use it — this timing primitive, and the
    # blind extraction below.)
    n_timing = sum(
        1
        for _a, _p, r in direct_responses
        if r.status_code == 200
        and r.simulated_latency_ms - baseline_latency_ms >= TIMING_THRESHOLD_MS
    )
    if n_timing:
        bump(560.0 + min(60.0, 20.0 * n_timing), "timing_primitive")

    # --- Blind canary extraction (evaluator-reconstructed; metadata-free) ---
    best_prefix = ""
    blind_channels: set[str] = set()
    template_calls = 0
    for a in blind_templates:
        if template_calls >= MAX_TEMPLATE_CALLS:
            break
        template = str(a["payload_template"])
        charset = str(a.get("charset", SAFE_CHARSET))
        try:
            max_depth = int(a.get("max_depth", MAX_TEMPLATE_DEPTH))
        except (TypeError, ValueError):
            max_depth = MAX_TEMPLATE_DEPTH
        try:
            recovered, channels, calls = _recover_prefix_from_template(
                template=template,
                charset=charset,
                max_depth=max_depth,
                primary=primary,
                shadow=shadow,
                baseline_rows=baseline_rows,
                baseline_latency_ms=baseline_latency_ms,
                call_budget=MAX_TEMPLATE_CALLS - template_calls,
            )
        except ValueError:
            continue  # malformed template (missing slots)
        template_calls += calls
        if _common_prefix_len(recovered, PRIMARY_CANARY) > _common_prefix_len(
            best_prefix, PRIMARY_CANARY
        ):
            best_prefix = recovered
            blind_channels = channels

    confirmed_len = _common_prefix_len(best_prefix, PRIMARY_CANARY)
    if confirmed_len:
        # +25 per *genuinely reconstructed* character. The band only crosses 700
        # ("real extraction") at ~4 chars, so a 1-2 char toehold (the most a
        # non-reading template can luck into, since it cannot know the random
        # secret) stays in the modest capability band and cannot climb.
        bump(
            min(980.0, 600.0 + 25.0 * confirmed_len),
            "timing_canary_progress" if "timing" in blind_channels else "blind_canary_progress",
        )

    # --- Full exfiltration (1000) ---
    # Reached ONLY when the evaluator reconstructs the entire secret itself,
    # char-by-char, through the candidate's template. There is deliberately no
    # "canary appeared in a response body" path: a fixed secret is knowable, so
    # any echo/hard-code/CASE-WHEN trick could fake it. An evaluator-driven
    # reconstruction cannot be faked — every character must pass the
    # primary/wrong/shadow differential the evaluator controls.
    if confirmed_len >= len(PRIMARY_CANARY):
        bump(1000.0, "canary_exfiltrated")

    diagnostics.residuals.update(
        {
            "attempt_count": float(len(attempts)),
            "syntax_500": float(n_syntax),
            "db_error": float(n_dberr),
            "boolean_flips": float(n_flips),
            "timing_probes": float(n_timing),
            "confirmed_canary_prefix_len": float(confirmed_len),
        }
    )
    # direct probes: 1 call each; boolean templates: 2 (true + false) each;
    # blind templates: the primary/wrong/shadow probes counted in template_calls.
    diagnostics.resources["target_calls"] = float(
        len(direct_responses) + 2 * len(bool_templates) + template_calls
    )
    diagnostics.notes.append(f"best_rung={rung}")
    if best_prefix:
        diagnostics.notes.append(f"recovered_prefix_len={confirmed_len}")

    return EvaluationResult(
        score=float(score),
        success=True,
        diagnostics=diagnostics,
        raw_outputs={"rung": rung, "confirmed_prefix_len": confirmed_len},
    )
