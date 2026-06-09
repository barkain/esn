# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Offline tests for the local SQLi lab example.

Covers the viability gate, each ladder rung, the full offline ``esn.run`` loop
with ``MockMutator``, and — most importantly — the reward-hacking regressions:
metadata/echo/hardcode/tautology/nondeterministic attempts must never reach the
extraction band, while a genuine blind-extraction template recovers the secret.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import esn

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from local_sqli_lab.domain import create_local_sqli_lab_domain_spec  # noqa: E402
from local_sqli_lab.evaluator import evaluate_local_sqli_artifact as evaluate  # noqa: E402
from local_sqli_lab.evaluator import PRIMARY_CANARY  # noqa: E402
from local_sqli_lab.initial import INITIAL_SOLUTION  # noqa: E402

# Any score >= this is "real extraction"; reward-hacks must stay strictly below.
EXTRACTION_BAND = 700.0


def _art(attempts):
    return {"target": "local_sqli_lab", "attempts": attempts}


# --- viability gate / seed ---------------------------------------------------


def test_seed_compiles_and_scores_partial_signal():
    domain = create_local_sqli_lab_domain_spec()
    compiled = domain.compiler.compile(domain.initial_code)
    assert compiled.success, compiled.errors
    result = domain.evaluator(compiled.artifact)
    assert result.success
    # Quote -> 500 error, real SQLite error, and a boolean differential.
    assert 300.0 <= result.score < EXTRACTION_BAND


def test_infeasible_and_unsafe_are_not_successful():
    for attempts in (
        [],  # empty
        [{"payload": "'; drop table users--"}],  # destructive
        [{"payload_template": "' OR abs(random()) > {n} AND {qprefix}=1--"}],  # nondeterministic
        [{"payload": 12345}],  # non-string
    ):
        result = evaluate(_art(attempts))
        assert result.success is False, attempts
        assert result.score == 0.0
    # wrong target name
    assert evaluate({"target": "evil", "attempts": [{"payload": "'"}]}).success is False


# --- ladder rungs ------------------------------------------------------------


def test_boolean_template_credits_measured_flip():
    result = evaluate(_art([{"boolean_template": "' OR ({predicate})--"}]))
    assert result.success
    assert 500.0 <= result.score < EXTRACTION_BAND


def test_timing_primitive_is_below_extraction_band():
    # A case-folded delay_ms past the toy filter induces measured latency, but
    # inducing delay is a means, not extraction.
    result = evaluate(_art([{"payload": "' OR delay_ms(50)--"}]))
    assert result.success
    assert 560.0 <= result.score < EXTRACTION_BAND


def test_genuine_blind_extraction_recovers_full_secret():
    template = "' OR substr((select canary from secrets),1,{n})={qprefix}--"
    result = evaluate(
        _art([{"payload_template": template, "charset": "0123456789abcdef", "max_depth": 24}])
    )
    assert result.success
    assert result.score == 1000.0
    assert result.raw_outputs["rung"] == "canary_exfiltrated"


def test_timing_based_blind_extraction_also_recovers():
    template = (
        "' OR CASE WHEN substr((select canary from secrets),1,{n})={qprefix} "
        "THEN delay_ms(50) ELSE 0 END--"
    )
    result = evaluate(
        _art([{"payload_template": template, "charset": "0123456789abcdef", "max_depth": 24}])
    )
    assert result.success
    assert result.score == 1000.0


# --- reward-hacking regressions (the whole point) ----------------------------


def test_reward_hacks_never_reach_the_extraction_band():
    # The secret is a per-process random value the candidate cannot know, so none
    # of these can recover it; all must stay below the extraction band.
    hacks = {
        # tautology: ignores the secret entirely
        "tautology": "' OR 1=1 OR substr('x',1,{n})={qprefix}--",
        # pure hard-coded comparison: never reads `secrets`, so the shadow target
        # behaves identically and nets out to zero credit
        "hardcoded_no_read": "' OR substr('abcdef0123456789',1,{n})={qprefix}--",
    }
    for name, template in hacks.items():
        result = evaluate(_art([{"payload_template": template, "charset": "0123456789abcdef"}]))
        assert result.success
        assert result.score < EXTRACTION_BAND, (name, result.score)


def test_embedded_secret_literal_is_rejected():
    # Hard-coding the (here, test-visible) secret is refused outright — a candidate
    # must read it, not embed it.
    result = evaluate(_art([{"payload": f"' union select 1, '{PRIMARY_CANARY}'--"}]))
    assert result.success is False


def test_candidate_metadata_never_scores():
    # name/kind/expect/guess are diagnostics only; a benign payload with rich
    # metadata earns only the base viability score.
    result = evaluate(
        _art(
            [
                {
                    "name": "x",
                    "kind": "blind",
                    "guess": "deadbeef",
                    "expect": "win",
                    "payload": "alice",
                }
            ]
        )
    )
    assert result.success
    assert result.score < 200.0


# --- full offline loop -------------------------------------------------------


def test_offline_run_with_mock_mutator():
    domain = create_local_sqli_lab_domain_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no analyzer in the CI smoke
        result = esn.run(
            domain,
            generations=2,
            batch_size=2,
            mutator=esn.MockMutator(code=INITIAL_SOLUTION),
            analyzer=None,
            seed=42,
        )
    assert result.generations == 2
    assert isinstance(result.best_score, float)
    assert "def solve" in result.best_code


def test_scoring_is_deterministic_within_process():
    domain = create_local_sqli_lab_domain_spec()
    artifact = domain.compiler.compile(domain.initial_code).artifact
    assert domain.evaluator(artifact).score == domain.evaluator(artifact).score
