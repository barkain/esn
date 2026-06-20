# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Non-gameable evaluator for the web CTF chain (path traversal -> JWT forge).

The evaluator owns *all* target interaction and scoring. Candidate ``solve()``
output is untrusted: every point is derived from **measured target behavior**
(response status/class), never from candidate metadata (``name``, ``note``, …).

Design (reviewed against the SQLi lab's non-gameable contract):

* **Independent-additive scoring.** Each stage has its own fixed, capped budget
  and is reachable on its own, so the scalar fitness gives dense partial credit —
  a good JWT recipe scores even if traversal is unsolved, and vice-versa. This is
  what keeps composable capabilities viable for ESN's branch/recombination
  machinery (a strictly gated chain would collapse all selection pressure onto
  stage 1). A *separate, smaller* ``chain_bonus`` rewards completing the real
  end-to-end kill chain.
* **Slots are attack surface.** The candidate's JWT recipe may reference exactly
  one evaluator-owned slot, ``{leaked_key}`` (the signing key a traversal would
  recover). It is substituted by an explicit replacer — never ``str.format`` —
  and any *other* ``{...}`` is rejected. Slot *inclusion* never scores; credit
  comes only when the primary app accepts the resulting request and the shadow
  app (different key) rejects it.
* **Nothing recovered is ever surfaced.** The signing key, the config body, and
  every constructed token are consumed internally. Diagnostics/notes/raw_outputs
  contain only numeric progress and non-secret behavioral *class* names, so a
  later generation can never read a recovered secret back out of the feedback
  loop and hard-code it.
* **Wrong-key + shadow controls.** Full JWT credit (``ok_admin``) requires the
  primary-signed token to verify on the primary and be *rejected* by the shadow,
  proving the forge depends on the real key, not a candidate constant.

Secrets are per-process CSPRNG hex, never written to source or the prompt.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Any

from esn import EvaluationDiagnostics, EvaluationResult

from .target import (
    LocalWebApp,
    C_BAD_ALG,
    C_BAD_SIGNATURE,
    C_ESCAPED_CONFIG,
    C_ESCAPED_FOUND,
    C_ESCAPED_NOTFOUND,
    C_EXPIRED,
    C_IN_BASE_FOUND,
    C_MISSING_EXP,
    C_OK_ADMIN,
    C_OK_ADMIN_AUD,
    C_OK_ADMIN_BADAUD,
    C_OK_ADMIN_BADSCOPE,
    C_OK_ADMIN_FRESH,
    C_OK_AUTHENTICATED,
    C_OK_GUEST,
    C_OK_ROOT,
    C_PARSE_ERROR,
)


# --- Per-process secrets (unknowable to candidates) --------------------------
def _new_key() -> str:
    return secrets.token_hex(16)  # 32 opaque hex chars


PRIMARY_KEY = _new_key()
SHADOW_KEY = _new_key()
while SHADOW_KEY == PRIMARY_KEY:
    SHADOW_KEY = _new_key()
_SECRETS = (PRIMARY_KEY, SHADOW_KEY)

# --- Budgets (independent-additive + chain bonus; UNCAPPED depth) ------------
# Each stage has its own capped budget (dense partial credit so ESN's search keeps
# half-working candidates viable); a separate chain bonus rewards the end-to-end
# kill chain. The JWT budget extends well ABOVE plain admin via the privilege
# ladder (admin -> +aud -> +freshness -> +root) so capability keeps paying off and
# stronger configs never plateau into a tie at "admin". The score is the raw sum;
# SCORE_MAX is only a diagnostic reference, never a clamp.
TRAVERSAL_MAX = 4000.0
JWT_MAX = 9000.0  # top rung (ok_root); plain admin is now a mid rung at 4500
CHAIN_BONUS = 1500.0
SCORE_MAX = TRAVERSAL_MAX + JWT_MAX + CHAIN_BONUS  # 14500 (reference, not a clamp)

# Traversal tiers (best reached wins; behavior-derived).
_TRAVERSAL_TIER = {
    C_IN_BASE_FOUND: 300.0,  # used the endpoint, read a public file
    C_ESCAPED_NOTFOUND: 1500.0,  # defeated the sanitizer (escaped base)
    C_ESCAPED_FOUND: 2500.0,  # escaped + hit a sensitive non-config file
    C_ESCAPED_CONFIG: 4000.0,  # reached the config file holding the key
}
# JWT tiers (best reached wins). The secret-dependent rungs (>= missing_exp) can
# only be reached when the HS256 signature verifies with the real key, i.e. the
# recipe used the {leaked_key} slot — a hard-coded key yields bad_signature.
_JWT_TIER = {
    C_PARSE_ERROR: 0.0,
    C_BAD_ALG: 200.0,  # structurally a JWT, wrong algorithm
    C_BAD_SIGNATURE: 400.0,  # chose HS256, reached the signature check
    C_OK_GUEST: 900.0,  # alg=none privilege-limited bypass (low rung)
    C_MISSING_EXP: 2300.0,  # signature VERIFIED with the real key; exp missing
    C_EXPIRED: 2300.0,  # signature verified; exp present but not in the future
    C_OK_AUTHENTICATED: 3400.0,  # verified + valid claims, role != admin
    C_OK_ADMIN: 4500.0,  # verified + valid claims + role == admin (mid rung)
    # Privilege ladder above admin — uncapped depth that separates capability:
    C_OK_ADMIN_AUD: 6000.0,  # admin + correct audience
    C_OK_ADMIN_FRESH: 7500.0,  # admin + aud + token freshness
    C_OK_ROOT: 9000.0,  # admin + aud + freshness + privileged scope (top)
    # "Present-but-wrong-value" rungs: a DISTINCT class for the feedback gradient,
    # scored the SAME as the rung below so a junk value earns no extra credit.
    C_OK_ADMIN_BADAUD: 4500.0,  # == C_OK_ADMIN
    C_OK_ADMIN_BADSCOPE: 7500.0,  # == C_OK_ADMIN_FRESH
}
# Rungs that assert dependence on the real signing key (need a shadow control).
_SECRET_DEPENDENT = {
    C_MISSING_EXP,
    C_EXPIRED,
    C_OK_AUTHENTICATED,
    C_OK_ADMIN,
    C_OK_ADMIN_AUD,
    C_OK_ADMIN_FRESH,
    C_OK_ROOT,
    C_OK_ADMIN_BADAUD,
    C_OK_ADMIN_BADSCOPE,
}
# Reaching role==admin at any ladder rung counts as "admin" for the chain bonus.
_ADMIN_OR_HIGHER = {
    C_OK_ADMIN,
    C_OK_ADMIN_AUD,
    C_OK_ADMIN_FRESH,
    C_OK_ROOT,
    C_OK_ADMIN_BADAUD,
    C_OK_ADMIN_BADSCOPE,
}

# Validation bounds.
MAX_TRAVERSAL_ATTEMPTS = 24
MAX_JWT_ATTEMPTS = 16
MAX_FIELD_LEN = 512
MAX_JSON_LEN = 1024
_KEY_SLOT = "{leaked_key}"


# ---------------------------------------------------------------------------
# Validation / coercion (the viability gate)
# ---------------------------------------------------------------------------
def _scan_strings(value: Any) -> list[str]:
    """Flatten every string contained in value (recursing dicts/lists)."""
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            out.extend(_scan_strings(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_scan_strings(v))
    elif value is not None:
        out.append(str(value))
    return out


def _bad_string(text: str) -> str | None:
    if len(text) > MAX_FIELD_LEN:
        return f"field too long ({len(text)} > {MAX_FIELD_LEN})"
    if any(s in text for s in _SECRETS):
        return "field embeds a secret literal (recover it, don't hard-code it)"
    return None


def _coerce(artifact: Any) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Return (traversal_paths, jwt_recipes, errors)."""
    errors: list[str] = []
    if not isinstance(artifact, dict):
        return [], [], ["solve() must return a dict artifact"]
    if artifact.get("target") != "web_ctf_chain":
        errors.append("artifact 'target' must be 'web_ctf_chain'")

    # Any embedded secret literal anywhere is a hard-coded answer -> reject.
    for text in _scan_strings(artifact):
        bad = _bad_string(text)
        if bad:
            return [], [], [bad]

    raw_trav = artifact.get("traversal") or []
    raw_jwt = artifact.get("jwt") or []
    if not isinstance(raw_trav, list) or not isinstance(raw_jwt, list):
        return [], [], ["'traversal' and 'jwt' must be lists"]
    if len(raw_trav) > MAX_TRAVERSAL_ATTEMPTS or len(raw_jwt) > MAX_JWT_ATTEMPTS:
        return [], [], ["too many attempts"]

    paths: list[str] = []
    seen_paths: set[str] = set()
    for item in raw_trav:
        path = (
            item
            if isinstance(item, str)
            else (item.get("path") if isinstance(item, dict) else None)
        )
        if not isinstance(path, str) or not path:
            errors.append("each traversal attempt needs a non-empty 'path' string")
            continue
        if path not in seen_paths:  # dedup: duplicates cannot farm
            seen_paths.add(path)
            paths.append(path)

    recipes: list[dict[str, Any]] = []
    seen_recipes: set[str] = set()
    for item in raw_jwt:
        if not isinstance(item, dict):
            errors.append("each jwt attempt must be a dict")
            continue
        header = item.get("header")
        claims = item.get("claims")
        # No default to the slot: an omitted key_slot must NOT silently resolve to
        # the real key. Absent -> empty literal -> the HMAC fails (bad_signature),
        # so a candidate must *explicitly* reference {leaked_key} to forge.
        key_slot = item.get("key_slot", "")
        if not isinstance(header, dict) or not isinstance(claims, dict):
            errors.append("jwt attempt needs 'header' and 'claims' objects")
            continue
        if not isinstance(key_slot, str):
            errors.append("jwt 'key_slot' must be a string")
            continue
        # Explicit slot policy: only {leaked_key} is recognized; any other brace
        # is rejected (no str.format, no unknown slots).
        if key_slot != _KEY_SLOT and ("{" in key_slot or "}" in key_slot):
            errors.append("jwt 'key_slot' contains an unknown slot")
            continue
        try:
            hj, cj = json.dumps(header), json.dumps(claims)
        except (TypeError, ValueError):
            errors.append("jwt header/claims must be JSON-serializable")
            continue
        if len(hj) > MAX_JSON_LEN or len(cj) > MAX_JSON_LEN:
            errors.append("jwt header/claims too large")
            continue
        sig = repr((hj, cj, key_slot))
        if sig not in seen_recipes:  # dedup
            seen_recipes.add(sig)
            recipes.append({"header": header, "claims": claims, "key_slot": key_slot})

    if not paths and not recipes and not errors:
        errors.append("no valid attempts")
    return paths, recipes, errors


# ---------------------------------------------------------------------------
# Token construction (evaluator-owned; the candidate never builds or sees one)
# ---------------------------------------------------------------------------
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _build_token(header: dict[str, Any], claims: dict[str, Any], key: str) -> str:
    h_b64 = _b64url(json.dumps(header).encode())
    p_b64 = _b64url(json.dumps(claims).encode())
    if header.get("alg") == "none":
        return f"{h_b64}.{p_b64}."
    mac = hmac.new(key.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256)
    return f"{h_b64}.{p_b64}.{_b64url(mac.digest())}"


def _resolve_key(key_slot: str) -> str:
    # Explicit replacer: only the recognized slot maps to the real key. Any other
    # value is used literally (and will fail the HMAC) — slots never auto-leak.
    return PRIMARY_KEY if key_slot == _KEY_SLOT else key_slot


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------
def evaluate_web_ctf_artifact(artifact: Any) -> EvaluationResult:
    diagnostics = EvaluationDiagnostics(
        constraints={
            "target": "web_ctf_chain_in_process",
            "max_traversal_attempts": float(MAX_TRAVERSAL_ATTEMPTS),
            "max_jwt_attempts": float(MAX_JWT_ATTEMPTS),
            "score_max": SCORE_MAX,
        }
    )

    paths, recipes, errors = _coerce(artifact)
    if errors:
        diagnostics.violations.extend(errors[:5])
        return EvaluationResult(score=0.0, success=False, diagnostics=diagnostics)

    primary = LocalWebApp(PRIMARY_KEY)
    shadow = LocalWebApp(SHADOW_KEY)
    target_calls = 0

    # --- Stage 1: path traversal (best behavioral class wins) ---------------
    trav_score = 0.0
    trav_class = "none"
    reached_config = False
    for path in paths:
        resp = primary.download(path)
        target_calls += 1
        tier = _TRAVERSAL_TIER.get(resp.klass, 0.0)
        if tier > trav_score:
            trav_score = tier
            trav_class = resp.klass
        if resp.klass == C_ESCAPED_CONFIG:
            reached_config = True

    # --- Stage 2: JWT forge (best behavioral class wins) --------------------
    jwt_score = 0.0
    jwt_class = "none"
    sig_verified = False
    admin = False
    for recipe in recipes:
        key = _resolve_key(recipe["key_slot"])
        token = _build_token(recipe["header"], recipe["claims"], key)
        primary_class = primary.api(token).klass
        target_calls += 1
        # Secret-dependent rungs (sig verified with the real key) require BOTH:
        #   1. the recipe explicitly used the {leaked_key} slot — so the "reference
        #      the recovered key" contract can't be skipped; and
        #   2. a fail-closed shadow control — the same token must be REJECTED as
        #      bad_signature by the shadow app (different key), proving the credit
        #      depends on the real secret, not a candidate-controlled constant.
        if primary_class in _SECRET_DEPENDENT:
            if recipe["key_slot"] != _KEY_SLOT:
                primary_class = C_BAD_SIGNATURE
            else:
                shadow_class = shadow.api(token).klass
                target_calls += 1
                if shadow_class != C_BAD_SIGNATURE:
                    primary_class = C_BAD_SIGNATURE
        tier = _JWT_TIER.get(primary_class, 0.0)
        if tier > jwt_score:
            jwt_score = tier
            jwt_class = primary_class
        if primary_class in _SECRET_DEPENDENT:
            sig_verified = True
        if primary_class in _ADMIN_OR_HIGHER:
            admin = True

    # --- Chain bonus: real end-to-end kill chain (redacted boolean) ---------
    full_chain = reached_config and admin
    chain = CHAIN_BONUS if full_chain else 0.0

    score = trav_score + jwt_score + chain

    # Redacted diagnostics: numeric progress + non-secret behavioral classes
    # only. No keys, tokens, config bodies, or expanded payloads ever appear.
    diagnostics.residuals.update(
        {
            "traversal_score": trav_score,
            "jwt_score": jwt_score,
            "chain_bonus": chain,
            "axis_traversal": round(trav_score / TRAVERSAL_MAX, 4),
            "axis_jwt": round(jwt_score / JWT_MAX, 4),
            "axis_chain": round(chain / CHAIN_BONUS, 4),
            "escaped_base": float(trav_score >= _TRAVERSAL_TIER[C_ESCAPED_NOTFOUND]),
            "reached_config": float(reached_config),
            "jwt_sig_verified": float(sig_verified),
            "jwt_admin": float(admin),
            "full_chain_complete": float(full_chain),
        }
    )
    diagnostics.resources["target_calls"] = float(target_calls)
    diagnostics.notes.append(f"traversal_class={trav_class}")
    diagnostics.notes.append(f"jwt_class={jwt_class}")
    diagnostics.notes.append(f"full_chain={full_chain}")

    return EvaluationResult(
        score=float(score),
        success=True,
        diagnostics=diagnostics,
        raw_outputs={
            "traversal_class": trav_class,
            "jwt_class": jwt_class,
            "full_chain": full_chain,
            "score": float(score),
        },
    )
