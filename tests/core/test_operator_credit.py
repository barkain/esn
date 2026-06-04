"""Tests for ESN core operator credit model."""

from __future__ import annotations

from esn.core.enums import SearchMode
from esn.core.models import OperatorStats
from esn.core.operator_credit import OperatorCreditModel


class TestOperatorCreditModel:
    def test_record_increments_attempts(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True)
        assert m.get_stats("op1").attempts == 1
        m.record("op1", compile_success=True, eval_success=False)
        assert m.get_stats("op1").attempts == 2

    def test_record_tracks_successes(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True)
        m.record("op1", compile_success=True, eval_success=False)
        m.record("op1", compile_success=False, eval_success=False)
        stats = m.get_stats("op1")
        assert stats.compile_successes == 2
        assert stats.eval_successes == 1

    def test_mean_score_delta(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, score_delta=1.0)
        m.record("op1", compile_success=True, eval_success=True, score_delta=3.0)
        assert abs(m.get_stats("op1").mean_score_delta - 2.0) < 1e-9

    def test_mean_epistemic_novelty(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, epistemic_novelty=0.4)
        m.record("op1", compile_success=True, eval_success=True, epistemic_novelty=0.6)
        assert abs(m.get_stats("op1").mean_epistemic_novelty - 0.5) < 1e-9

    def test_get_stats_unknown_returns_default(self):
        m = OperatorCreditModel()
        stats = m.get_stats("nonexistent")
        assert stats.attempts == 0
        assert isinstance(stats, OperatorStats)

    def test_get_all_stats(self):
        m = OperatorCreditModel()
        m.record("a", compile_success=True, eval_success=True)
        m.record("b", compile_success=True, eval_success=False)
        all_stats = m.get_all_stats()
        assert len(all_stats) == 2
        assert "a" in all_stats
        assert "b" in all_stats

    def test_sample_returns_eligible(self):
        m = OperatorCreditModel()
        result = m.sample_operator(["a", "b", "c"], SearchMode.EXPLOIT)
        assert result in {"a", "b", "c"}

    def test_sample_empty_raises(self):
        m = OperatorCreditModel()
        try:
            m.sample_operator([], SearchMode.EXPLOIT)
            assert False, "Should have raised"  # noqa: B011
        except ValueError:
            pass

    def test_sample_ucb_exploration_bonus(self):
        m = OperatorCreditModel()
        # op_a has many attempts, op_b has few — UCB should favor exploring op_b
        for _ in range(20):
            m.record("op_a", compile_success=True, eval_success=True, score_delta=0.1)
        m.record("op_b", compile_success=True, eval_success=True, score_delta=0.1)
        result = m.sample_operator(["op_a", "op_b"], SearchMode.EXPLOIT, exploration_rate=0.0)
        assert result == "op_b"

    def test_sample_exploit_prefers_high_score_delta(self):
        m = OperatorCreditModel()
        for _ in range(10):
            m.record("good", compile_success=True, eval_success=True, score_delta=5.0)
            m.record("bad", compile_success=True, eval_success=True, score_delta=0.01)
        result = m.sample_operator(["good", "bad"], SearchMode.EXPLOIT, exploration_rate=0.0)
        assert result == "good"

    def test_sample_explore_prefers_high_novelty(self):
        m = OperatorCreditModel()
        for _ in range(10):
            m.record("novel", compile_success=True, eval_success=True, epistemic_novelty=5.0)
            m.record("boring", compile_success=True, eval_success=True, epistemic_novelty=0.01)
        result = m.sample_operator(["novel", "boring"], SearchMode.EXPLORE, exploration_rate=0.0)
        assert result == "novel"

    def test_to_dict_from_dict_roundtrip(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, score_delta=1.5, generation=3)
        m.record("op2", compile_success=False, eval_success=False, score_delta=-0.5)
        data = m.to_dict()
        m2 = OperatorCreditModel.from_dict(data)
        assert m2.get_stats("op1").attempts == 1
        assert abs(m2.get_stats("op1").mean_score_delta - 1.5) < 1e-9
        assert m2.get_stats("op2").compile_successes == 0

    def test_last_used_generation(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, generation=5)
        m.record("op1", compile_success=True, eval_success=True, generation=3)
        assert m.get_stats("op1").last_used_generation == 5

    def test_non_improving_streak_tracks_valid_no_gain(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, score_delta=0.0)
        m.record("op1", compile_success=True, eval_success=True, score_delta=-0.1)
        assert m.get_stats("op1").non_improving_streak == 2

    def test_improvement_resets_non_improving_streak(self):
        m = OperatorCreditModel()
        m.record("op1", compile_success=True, eval_success=True, score_delta=0.0)
        m.record("op1", compile_success=True, eval_success=True, score_delta=1.0)
        assert m.get_stats("op1").non_improving_streak == 0

    def test_recent_score_delta_biases_selection(self):
        m = OperatorCreditModel()
        for _ in range(10):
            m.record("stale", compile_success=True, eval_success=True, score_delta=0.0)
        for delta in [0.0] * 9 + [1.0]:
            m.record("fresh", compile_success=True, eval_success=True, score_delta=delta)
        result = m.sample_operator(["stale", "fresh"], SearchMode.EXPLOIT, exploration_rate=0.0)
        assert result == "fresh"
