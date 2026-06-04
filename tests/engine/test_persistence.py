"""Tests for ESN engine save/load persistence."""

from __future__ import annotations

from esn.core.models import EvaluationResult
from esn.engine.compiler import PythonSandboxCompiler
from esn.engine.domain import DomainSpec
from esn.engine.engine import ESNEngine


INITIAL_CODE = "def solve():\n    return [1, 2, 3]\n"


def _sum_evaluator(artifact):
    if artifact is None:
        return EvaluationResult(score=0.0, success=False)
    try:
        return EvaluationResult(score=float(sum(artifact)), success=True)
    except Exception:
        return EvaluationResult(score=0.0, success=False)


def _make_domain():
    return DomainSpec(
        name="persist-test",
        description="test domain for persistence",
        initial_code=INITIAL_CODE,
        compiler=PythonSandboxCompiler(allowed_imports=frozenset({"math"})),
        evaluator=_sum_evaluator,
    )


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        engine = ESNEngine(domain=_make_domain())
        for _ in range(3):
            engine.run_generation()

        save_dir = tmp_path / "checkpoint"
        engine.save_state(save_dir)

        engine2 = ESNEngine(domain=_make_domain())
        engine2.load_state(save_dir)

        assert engine2.generation == engine.generation
        assert engine2._best_score == engine._best_score
        assert engine2._best_code == engine._best_code
        assert engine2.elite_archive.size == engine.elite_archive.size

    def test_save_creates_expected_files(self, tmp_path):
        engine = ESNEngine(domain=_make_domain())
        engine.run_generation()

        save_dir = tmp_path / "checkpoint"
        engine.save_state(save_dir)

        expected_files = [
            "search_state.json",
            "elite.json",
            "frontier.json",
            "credit.json",
            "v3_state.json",
            "programs.json",
        ]
        for fname in expected_files:
            assert (save_dir / fname).exists(), f"Missing file: {fname}"

    def test_load_partial_state(self, tmp_path):
        engine = ESNEngine(domain=_make_domain())
        engine.run_generation()

        save_dir = tmp_path / "checkpoint"
        engine.save_state(save_dir)

        # Remove optional files, keep only search_state.json and v3_state.json
        for fname in ["elite.json", "frontier.json", "credit.json", "programs.json"]:
            f = save_dir / fname
            if f.exists():
                f.unlink()

        engine2 = ESNEngine(domain=_make_domain())
        # Should not crash
        engine2.load_state(save_dir)
        assert engine2.generation == engine.generation

    def test_seen_hashes_persisted(self, tmp_path):
        engine = ESNEngine(domain=_make_domain())
        engine.run_generation()

        assert len(engine._seen_hashes) >= 1
        original_hashes = set(engine._seen_hashes)

        save_dir = tmp_path / "checkpoint"
        engine.save_state(save_dir)

        engine2 = ESNEngine(domain=_make_domain())
        engine2.load_state(save_dir)

        assert engine2._seen_hashes == original_hashes
