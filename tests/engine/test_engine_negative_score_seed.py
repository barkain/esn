"""Regression tests: engine must handle negative scores correctly.

Covers two related latent defects:

1. ``ESNEngine._evaluate_seed_if_needed`` silently rejected the seed when
   the evaluator reported a non-positive score. That pathology breaks
   loss-shaped benchmarks (e.g. ``benchmarks/research_loop``) which emit
   ``score = -val_loss``: ``_best_score`` never departs from the 0.0 initial
   value, ``_best_code`` never updates, and every generation's mutator is
   fed the initial code — branch preservation becomes vacuous.

2. ``ESNEngine._finalize_batch`` used a multiplicative breakthrough
   deadband ``score > _best_score * 1.005`` that inverts sign for negative
   baselines. A baseline of ``-50.0`` would accept any score above
   ``-50.25`` — i.e. scores *worse* than the baseline would register as
   "improvements." The batched path (``_finalize_batch``) is the primary
   path for ``batch_size>=1`` runs, so the fix at the single-generation
   site (line 1647) is not sufficient on its own.

Both tests must fail on unfixed code and pass after the corresponding fix.
"""

from __future__ import annotations

import json

from esn.core.enums import SearchMode
from esn.core.models import (
    CompilerResult,
    EvaluationDiagnostics,
    EvaluationResult,
)
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine, _CandidateOutcome


SEED_CODE = """\
def solve():
    return [1, 2, 3]
"""


class _AlwaysCompileCompiler:
    """Minimal ProgramCompiler stub: compiles any source to a passthrough artifact."""

    def compile(self, source: str) -> CompilerResult:
        return CompilerResult(artifact=source, success=True)


def _negative_score_evaluator(_artifact: object) -> EvaluationResult:
    return EvaluationResult(
        score=-1.0,
        success=True,
        diagnostics=EvaluationDiagnostics(),
    )


def test_seed_eval_accepts_negative_score() -> None:
    """Seed eval must record a successful negative-score evaluation.

    Prior to the fix:
      - Line 267 gated seeding on ``_best_score > 0`` (harmless here but
        conceptually wrong for negative baselines).
      - Line 275 rejected the eval when ``eval_result.score <= 0`` even though
        the evaluator reported ``success=True``. This is what caused
        ``_best_score`` to remain at 0.0 in research_loop runs.

    After the fix, ``_best_score`` should equal the evaluator's reported
    score, ``_seed_evaluated`` should be True, and the elite archive should
    carry the seed record.
    """
    domain = DomainSpec(
        name="neg-seed-toy",
        description="toy domain returning negative eval scores",
        initial_code=SEED_CODE,
        compiler=_AlwaysCompileCompiler(),
        evaluator=_negative_score_evaluator,
    )
    engine = ESNEngine(domain=domain)

    engine._evaluate_seed_if_needed()

    assert engine._best_score == -1.0, (
        f"Seed eval with score=-1.0 should set _best_score=-1.0, "
        f"got {engine._best_score!r}. This indicates the seed gate silently "
        f"rejected the negative score."
    )
    assert getattr(engine, "_seed_evaluated", False) is True, (
        "Engine should expose a _seed_evaluated flag set to True after a "
        "successful seed evaluation; the flag is required so the seed gate "
        "does not re-fire after a negative-score seed."
    )
    assert engine.elite_archive.size >= 1, (
        "Elite archive must contain the seed candidate after a successful negative-score seed eval."
    )
    assert engine._best_code == SEED_CODE, (
        "_best_code should still reference the initial seed source."
    )


def _make_outcome(score: float, new_code: str) -> _CandidateOutcome:
    """Build a minimal _CandidateOutcome for _finalize_batch input."""
    return _CandidateOutcome(
        slot=0,
        style="refine",
        mode=SearchMode.EXPLOIT,
        parent_code=SEED_CODE,
        success=True,
        score=score,
        raw_score=score,
        new_code=new_code,
        eval_result=EvaluationResult(
            score=score,
            success=True,
            diagnostics=EvaluationDiagnostics(),
        ),
    )


def test_batched_incumbent_update_additive_deadband() -> None:
    """``_finalize_batch`` must use a sign-agnostic additive deadband.

    Prior to the fix at line 1265, the incumbent-update check was
    ``best_outcome.score > self._best_score * 1.005``. With
    ``_best_score = -50.0`` that expands to
    ``score > -50.0 * 1.005 = -50.25`` — i.e. *any* score above -50.25 would
    be accepted as "improvement", including -50.2 which is strictly worse
    than the -50.0 baseline.

    After the fix, the additive deadband ``-50.0 + max(50.0, 1.0) * 0.005 =
    -49.75`` must:
      - ACCEPT score=-49.0 (a real improvement past the deadband)
      - REJECT score=-50.2 (strictly worse than baseline)
      - REJECT score=-49.8 (inside the deadband, no meaningful improvement)
    """
    domain = DomainSpec(
        name="neg-batch-toy",
        description="toy domain for batched incumbent-update deadband test",
        initial_code=SEED_CODE,
        compiler=_AlwaysCompileCompiler(),
        evaluator=_negative_score_evaluator,
    )
    engine = ESNEngine(domain=domain)

    # Seed the engine's incumbent state directly, bypassing _evaluate_seed_if_needed
    # so the test focuses on the _finalize_batch deadband in isolation.
    engine._best_score = -50.0
    engine._best_code = SEED_CODE

    # Case 1: strictly worse than baseline must be rejected.
    worse = _make_outcome(score=-50.2, new_code="def solve():\n    return 'worse'\n")
    engine._finalize_batch(outcomes=[worse], any_success=True, best_outcome=worse)
    assert engine._best_score == -50.0, (
        f"Score -50.2 is strictly worse than baseline -50.0 and must NOT "
        f"update the incumbent. _best_score={engine._best_score!r}. This is "
        f"the core sign-inversion defect at _finalize_batch."
    )
    assert engine._best_code == SEED_CODE, (
        "_best_code must not change when the incumbent update is rejected."
    )

    # Case 2: inside the 0.5% * |baseline| deadband must be rejected.
    inside = _make_outcome(score=-49.8, new_code="def solve():\n    return 'inside'\n")
    engine._finalize_batch(outcomes=[inside], any_success=True, best_outcome=inside)
    assert engine._best_score == -50.0, (
        f"Score -49.8 is inside the 0.5% deadband around -50.0 (threshold "
        f"-49.75) and must NOT update the incumbent. "
        f"_best_score={engine._best_score!r}."
    )

    # Case 3: clear improvement past the deadband must be accepted.
    better_code = "def solve():\n    return 'better'\n"
    better = _make_outcome(score=-49.0, new_code=better_code)
    engine._finalize_batch(outcomes=[better], any_success=True, best_outcome=better)
    assert engine._best_score == -49.0, (
        f"Score -49.0 clears the -49.75 additive deadband and must update "
        f"the incumbent. _best_score={engine._best_score!r}."
    )
    assert engine._best_code == better_code, (
        "_best_code must update to the new code when the incumbent is accepted."
    )


def _build_engine() -> ESNEngine:
    """Construct a fresh engine with the negative-score domain wiring."""
    domain = DomainSpec(
        name="neg-seed-toy",
        description="toy domain returning negative eval scores",
        initial_code=SEED_CODE,
        compiler=_AlwaysCompileCompiler(),
        evaluator=_negative_score_evaluator,
    )
    return ESNEngine(domain=domain)


def test_seed_evaluated_persists_across_save_load(tmp_path) -> None:
    """``_seed_evaluated`` must round-trip through save_state/load_state.

    Without persistence, a resumed run on a loss-shaped benchmark would
    re-enter ``_evaluate_seed_if_needed`` with ``_seed_evaluated=False`` on
    the restored engine. The elite-archive side-channel happens to cover
    most cases, but if the archive ever loads empty (or the seed gate is
    consulted before the archive is restored), the gate would silently
    re-execute the seed eval. Persisting the explicit flag makes the
    invariant robust.

    Test plan:
      1. Build engine, seed it via ``_evaluate_seed_if_needed``.
      2. Verify ``_seed_evaluated is True`` and ``_best_score == -1.0``.
      3. Save to tmp_path, build a fresh engine, load from tmp_path.
      4. Assert restored engine has ``_seed_evaluated is True`` and
         ``_best_score == -1.0``.
      5. Round-trip the v3_state.json payload through ``json`` to confirm
         the field serializes natively.
      6. Backward compat: load a v3_state.json with the key stripped and
         confirm ``_seed_evaluated`` defaults to False without raising.
    """
    # --- 1+2: seed an engine ---
    engine = _build_engine()
    engine._evaluate_seed_if_needed()
    assert engine._seed_evaluated is True
    assert engine._best_score == -1.0

    # --- 3: round-trip through save_state/load_state ---
    save_dir = tmp_path / "snapshot"
    engine.save_state(save_dir)

    restored = _build_engine()
    # Sanity: a fresh engine starts un-seeded.
    assert restored._seed_evaluated is False
    assert restored._best_score == 0.0

    restored.load_state(save_dir)

    # --- 4: assertions on restored state ---
    assert restored._best_score == -1.0, (
        f"After load, _best_score should be -1.0, got {restored._best_score!r}. "
        f"This indicates the negative seed score didn't survive the round-trip."
    )
    assert restored._seed_evaluated is True, (
        f"After load, _seed_evaluated should be True, got "
        f"{restored._seed_evaluated!r}. The flag must round-trip through "
        f"save_state/load_state so the seed gate doesn't re-fire on resume."
    )

    # --- 5: confirm the field is present in the on-disk JSON payload ---
    v3_payload = json.loads((save_dir / "v3_state.json").read_text())
    assert "seed_evaluated" in v3_payload, "v3_state.json must serialize the seed_evaluated key."
    assert v3_payload["seed_evaluated"] is True

    # --- 6: backward compat — strip the key, confirm default-False load ---
    legacy_payload = {k: v for k, v in v3_payload.items() if k != "seed_evaluated"}
    (save_dir / "v3_state.json").write_text(json.dumps(legacy_payload))

    legacy_restored = _build_engine()
    legacy_restored.load_state(save_dir)
    assert legacy_restored._seed_evaluated is False, (
        "Loading a legacy snapshot without the seed_evaluated key must "
        "default the flag to False without raising."
    )
    # Sanity: the rest of the v3 payload still loaded cleanly.
    assert legacy_restored._best_score == -1.0, (
        "Legacy snapshot load must still restore _best_score."
    )
