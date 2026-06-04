"""Tests for ESN core search mode selector."""

from __future__ import annotations

from esn.core.enums import SearchMode
from esn.core.models import SearchState
from esn.core.search_mode_selector import SearchModeSelector


class TestSearchModeSelector:
    def setup_method(self):
        self.selector = SearchModeSelector()

    def test_default_is_exploit(self):
        state = SearchState()
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_high_stagnation_even_gen_explore(self):
        state = SearchState(
            stagnation_counter=9,
            generation=10,
            recent_operators=["swap"] * 8,
            frontier_distinct_count=3,
        )
        assert self.selector.select_mode(state) == SearchMode.EXPLORE

    def test_high_stagnation_without_diversity_signal_stays_exploit(self):
        state = SearchState(
            stagnation_counter=9, generation=11, recent_operators=["a", "b", "c", "d"]
        )
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_moderate_stagnation_exploit(self):
        state = SearchState(stagnation_counter=4)
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_high_failure_rate_repair(self):
        state = SearchState(recent_scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        # 8/10 failures (scores <= 0) = 0.8 failure rate
        assert self.selector.select_mode(state) == SearchMode.REPAIR

    def test_moderate_failure_with_stagnation_recover(self):
        state = SearchState(
            recent_scores=[0.0, 0.0, 0.0, 1.0, 1.0],  # 3/5 = 0.6 > 0.5
            stagnation_counter=2,
        )
        assert self.selector.select_mode(state) == SearchMode.RECOVER

    def test_improving_scores_exploit(self):
        state = SearchState(recent_scores=[0.5, 0.7, 0.9])
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_spectral_complexity_compress(self):
        state = SearchState(recent_scores=[0.5, 0.5, 0.5])
        summary = {"complexity_trend": 0.5}
        assert self.selector.select_mode(state, spectral_summary=summary) == SearchMode.COMPRESS

    def test_no_spectral_summary_default(self):
        state = SearchState(recent_scores=[0.5, 0.5, 0.5])
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_stagnation_takes_priority_over_failure(self):
        state = SearchState(
            stagnation_counter=9,
            generation=10,
            recent_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
            recent_operators=["swap"] * 8,
            frontier_distinct_count=3,
        )
        # High failure repair still wins over explore.
        assert self.selector.select_mode(state) == SearchMode.REPAIR

    def test_explore_requires_distinct_frontier(self):
        state = SearchState(
            stagnation_counter=9,
            recent_operators=["swap"] * 8,
            frontier_distinct_count=1,
        )
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_explore_requires_low_operator_diversity(self):
        state = SearchState(
            stagnation_counter=9,
            recent_operators=["a", "b", "c", "d", "e", "f", "g", "h"],
            frontier_distinct_count=4,
        )
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_failure_rate_below_threshold_no_repair(self):
        state = SearchState(recent_scores=[1.0, 1.0, 1.0, 0.0])  # 0.25 failure
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT

    def test_empty_scores_default(self):
        state = SearchState(recent_scores=[])
        assert self.selector.select_mode(state) == SearchMode.EXPLOIT
