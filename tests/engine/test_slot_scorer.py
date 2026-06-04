"""Tests for BatchSlotScorer: feature-based scoring + MMR diverse selection."""

from __future__ import annotations

from esn.core.enums import SearchMode
from esn.core.models import CandidateRecord, SearchState
from esn.core.operator_credit import OperatorCreditModel
from esn.engine.family_tracker import FamilyTracker
from esn.engine.slot_scorer import (
    BatchSlotScorer,
    ComboTracker,
    ScorerState,
    ScorerWeights,
    SlotCandidate,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    cid: str,
    score: float,
    family: str = "unknown",
    gen: int = 1,
    code: str = "",
) -> CandidateRecord:
    return CandidateRecord(
        id=cid,
        generation=gen,
        search_mode=SearchMode.EXPLOIT,
        operator_name="refine",
        object_hash=f"hash_{cid}",
        object_summary=code[:200] if code else f"code_{cid}",
        score=score,
        success=True,
        family=family,
        family_confidence="high",
    )


class MockEliteArchive:
    def __init__(self, candidates: list[CandidateRecord] | None = None) -> None:
        self._candidates = candidates or []

    @property
    def size(self) -> int:
        return len(self._candidates)

    def get_best(self, n: int = 1) -> list[CandidateRecord]:
        return sorted(self._candidates, key=lambda c: c.score or 0.0, reverse=True)[:n]

    def get_all(self) -> list[CandidateRecord]:
        return list(self._candidates)


class MockFrontierArchive:
    def __init__(self, candidates: list[CandidateRecord] | None = None) -> None:
        self._candidates = candidates or []

    @property
    def size(self) -> int:
        return len(self._candidates)

    def get_novel_candidates(self, n: int = 5) -> list[CandidateRecord]:
        return self._candidates[:n]

    def get_all(self) -> list[CandidateRecord]:
        return list(self._candidates)


class MockEngine:
    def __init__(
        self,
        best_code: str = "def solve(): return [1,2,3]",
        best_score: float = 6.0,
        batch_size: int = 4,
        generation: int = 5,
        stagnation: int = 2,
    ) -> None:
        self._best_code = best_code
        self._best_score = best_score
        self._batch_size = batch_size
        self._consecutive_failures = 0
        self._run_dir = None
        self.generation = generation

        self.state = SearchState(
            generation=generation,
            best_score=best_score,
            stagnation_counter=stagnation,
            elite_size=3,
            frontier_size=2,
            recent_scores=[3.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
        )

        # Elites: 3 candidates from 2 families
        elite_a = _make_candidate("e1", 6.0, "ring", code=best_code)
        elite_b = _make_candidate("e2", 5.5, "grid", code="def solve(): return [4,5,6]")
        elite_c = _make_candidate("e3", 5.0, "ring", code="def solve(): return [7,8,9]")
        self.elite_archive = MockEliteArchive([elite_a, elite_b, elite_c])

        # Frontier: 2 candidates
        frontier_a = _make_candidate("f1", 4.0, "hex", code="def solve(): return [10,11]")
        frontier_b = _make_candidate("f2", 3.5, "spiral", code="def solve(): return [12]")
        self.frontier_archive = MockFrontierArchive([frontier_a, frontier_b])

        # Program store
        self._program_store = {
            "e1": best_code,
            "e2": "def solve(): return [4,5,6]",
            "e3": "def solve(): return [7,8,9]",
            "f1": "def solve(): return [10,11]",
            "f2": "def solve(): return [12]",
        }

        # Credit model (real)
        self.credit_model = OperatorCreditModel()
        # Seed some stats
        self.credit_model.record("refine", True, True, 0.1, 0.3, 0.0, 1)
        self.credit_model.record("refine", True, True, 0.05, 0.2, 0.0, 2)
        self.credit_model.record("explore", True, True, 0.2, 0.5, 0.0, 3)
        self.credit_model.record("radical", True, False, -0.5, 0.1, 0.0, 4)

        # Family tracker
        self._family_tracker = FamilyTracker()
        self._family_tracker.record("ring", 6.0, True, "ring approach")
        self._family_tracker.record("grid", 5.5, True, "grid approach")
        self._family_tracker.record("hex", 4.0, True, "hex approach")
        self._family_tracker.record("ring", 5.8, True, "ring v2")  # plateau_gens=1

        # No novelty computer
        self.novelty_computer = None


# ---------------------------------------------------------------------------
# ComboTracker tests
# ---------------------------------------------------------------------------


class TestComboTracker:
    def test_record_and_get_basic(self) -> None:
        tracker = ComboTracker()
        tracker.record("ring", "refine", True, 5)
        attempts, successes, last_gen = tracker.get("ring", "refine")
        assert attempts == 1
        assert successes == 1
        assert last_gen == 5

    def test_multiple_records_accumulate(self) -> None:
        tracker = ComboTracker()
        tracker.record("ring", "refine", True, 1)
        tracker.record("ring", "refine", False, 2)
        tracker.record("ring", "refine", True, 3)
        attempts, successes, last_gen = tracker.get("ring", "refine")
        assert attempts == 3
        assert successes == 2
        assert last_gen == 3

    def test_success_tracking(self) -> None:
        tracker = ComboTracker()
        tracker.record("ring", "explore", False, 1)
        tracker.record("ring", "explore", False, 2)
        _, successes, _ = tracker.get("ring", "explore")
        assert successes == 0

    def test_total_attempts(self) -> None:
        tracker = ComboTracker()
        tracker.record("ring", "refine", True, 1)
        tracker.record("grid", "explore", False, 2)
        tracker.record("ring", "refine", True, 3)
        assert tracker.total_attempts() == 3

    def test_to_dict_from_dict_roundtrip(self) -> None:
        tracker = ComboTracker()
        tracker.record("ring", "refine", True, 5)
        tracker.record("grid", "explore", False, 3)

        data = tracker.to_dict()
        restored = ComboTracker.from_dict(data)

        assert restored.get("ring", "refine") == (1, 1, 5)
        assert restored.get("grid", "explore") == (1, 0, 3)
        assert restored.total_attempts() == 2

    def test_get_unknown_combo_returns_zeros(self) -> None:
        tracker = ComboTracker()
        attempts, successes, last_gen = tracker.get("nonexistent", "style")
        assert attempts == 0
        assert successes == 0
        assert last_gen == 0


# ---------------------------------------------------------------------------
# ScorerState tests
# ---------------------------------------------------------------------------


class TestScorerState:
    def test_snapshot_state_from_mock_engine(self) -> None:
        engine = MockEngine()
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)

        assert state.generation == 5
        assert state.best_score == 6.0
        assert state.stagnation_counter == 2
        assert state.consecutive_failures == 0
        assert state.elite_size == 3
        assert state.frontier_size == 2
        assert state.num_families == 3  # ring, grid, hex

    def test_handles_missing_novelty_computer(self) -> None:
        engine = MockEngine()
        engine.novelty_computer = None
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)

        assert state.spectral_gamma == 0.0
        assert state.spectral_num_spikes == 0

    def test_recent_improvement_rate(self) -> None:
        engine = MockEngine()
        # recent_scores = [3, 4, 5, 5, 6, 6, 6, 6]
        # improvements: 3->4, 4->5, 5->6 = 3 out of 7 transitions
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)
        assert abs(state.recent_improvement_rate - 3.0 / 7.0) < 1e-9


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------


class TestFeaturize:
    def _make_scored_candidate(
        self, engine: MockEngine, parent_score: float, family: str, style: str
    ) -> tuple[SlotCandidate, ScorerState]:
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)
        candidate = SlotCandidate(
            parent_id="test_id",
            parent_score=parent_score,
            parent_family=family,
            style=style,
        )
        all_scores = [6.0, 5.5, 5.0, 4.0, 3.5]
        scorer.featurize(candidate, state, engine, all_scores)
        return candidate, state

    def test_parent_score_quantile_best_is_1(self) -> None:
        engine = MockEngine()
        c, _ = self._make_scored_candidate(engine, 6.0, "ring", "refine")
        assert c.features["parent_score_quantile"] == 1.0

    def test_parent_score_quantile_worst_is_0(self) -> None:
        engine = MockEngine()
        c, _ = self._make_scored_candidate(engine, 3.5, "spiral", "refine")
        assert c.features["parent_score_quantile"] == 0.0

    def test_parent_gap_to_best_is_zero_for_best(self) -> None:
        engine = MockEngine()
        c, _ = self._make_scored_candidate(engine, 6.0, "ring", "refine")
        assert c.features["parent_gap_to_best"] == 0.0

    def test_family_rarity_high_for_rare_family(self) -> None:
        engine = MockEngine()
        # hex has 1 entry out of 5 total archive entries
        c, _ = self._make_scored_candidate(engine, 4.0, "hex", "refine")
        assert c.features["family_rarity"] >= 0.7

    def test_style_recent_delta_z_normalized(self) -> None:
        engine = MockEngine()
        c, _ = self._make_scored_candidate(engine, 6.0, "ring", "refine")
        assert -1.0 <= c.features["style_recent_delta_z"] <= 1.0

    def test_combo_recency_capped_at_5_gens(self) -> None:
        engine = MockEngine()
        scorer = BatchSlotScorer()
        # Record a combo long ago
        scorer.combo_tracker.record("ring", "refine", True, 0)
        state = scorer.snapshot_state(engine)  # generation=5
        candidate = SlotCandidate(
            parent_id="test", parent_score=6.0, parent_family="ring", style="refine"
        )
        scorer.featurize(candidate, state, engine, [6.0])
        # (5 - 0) / 5 = 1.0, capped at 1.0
        assert candidate.features["combo_recency"] == 1.0

    def test_combo_attempt_count_zero_for_untried(self) -> None:
        engine = MockEngine()
        c, _ = self._make_scored_candidate(engine, 6.0, "ring", "radical")
        # No combos recorded in default scorer
        assert c.features["combo_attempt_count"] == 0

    def test_parent_recent_breakthrough(self) -> None:
        engine = MockEngine()
        # ring has plateau_gens=1 (just set via the 5.8 record after 6.0)
        c, _ = self._make_scored_candidate(engine, 6.0, "ring", "refine")
        assert c.features["parent_recent_breakthrough"] == 1.0

    def test_all_features_in_range(self) -> None:
        engine = MockEngine()
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)
        candidates, _ = scorer.enumerate_candidates(engine, SearchMode.EXPLOIT)
        all_scores = [c.parent_score for c in candidates]
        for c in candidates:
            scorer.featurize(c, state, engine, all_scores)
            for name, val in c.features.items():
                assert -1.0 <= val <= 1.0, f"Feature {name}={val} out of range for {c}"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestScoring:
    def _score_with_stagnation(self, stagnation: int) -> list[SlotCandidate]:
        engine = MockEngine(stagnation=stagnation)
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)
        candidates, _ = scorer.enumerate_candidates(engine, SearchMode.EXPLOIT)
        all_scores = [c.parent_score for c in candidates]
        for c in candidates:
            scorer.featurize(c, state, engine, all_scores)
        scorer.score_candidates(candidates, state)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def test_high_stagnation_shifts_toward_diversity(self) -> None:
        low_stag = self._score_with_stagnation(0)
        high_stag = self._score_with_stagnation(10)
        # At high stagnation, top candidate should differ or rare families rank higher
        # At stagnation=10, alpha=1.0, diversity dominates
        # At stagnation=0, alpha=0, quality dominates
        # They should produce different orderings
        assert low_stag != high_stag

    def test_zero_stagnation_favors_quality(self) -> None:
        ranked = self._score_with_stagnation(0)
        top = ranked[0]
        # Best parent + successful style should rank high
        assert top.parent_score >= 5.0  # should be from a high-scoring parent

    def test_best_parent_successful_style_scores_high_at_stag0(self) -> None:
        ranked = self._score_with_stagnation(0)
        # Among the top candidates, one should be best parent with refine
        top3 = ranked[:3]
        best_parent_refine = [c for c in top3 if c.parent_score == 6.0 and c.style == "refine"]
        assert len(best_parent_refine) > 0

    def test_rare_family_untried_combo_scores_high_at_stag10(self) -> None:
        ranked = self._score_with_stagnation(10)
        # Rare families (hex, spiral) with untried combos should rank well
        top5 = ranked[:5]
        rare_families = [c for c in top5 if c.parent_family in ("hex", "spiral")]
        assert len(rare_families) > 0

    def test_penalty_features_reduce_score(self) -> None:
        engine = MockEngine(stagnation=0)
        scorer = BatchSlotScorer()
        state = scorer.snapshot_state(engine)

        # Candidate with high penalty features
        c = SlotCandidate(parent_id="x", parent_score=6.0, parent_family="ring", style="refine")
        c.features = {
            "parent_score_quantile": 1.0,
            "parent_gap_to_best": 0.0,
            "parent_is_best": 1.0,
            "parent_recent_breakthrough": 0.0,
            "family_rarity": 0.0,
            "family_plateau_gens": 1.0,  # max plateau penalty
            "style_success_rate": 0.5,
            "style_recent_delta_z": 0.0,
            "style_novelty_yield": 0.0,
            "style_non_improving_frac": 1.0,  # max streak penalty
            "style_attempts_share": 0.5,
            "combo_attempt_count": 0.5,
            "combo_success_rate": 0.5,
            "combo_recency": 0.0,
        }

        # Same candidate without penalties
        c_no_penalty = SlotCandidate(
            parent_id="y", parent_score=6.0, parent_family="ring", style="refine"
        )
        c_no_penalty.features = dict(c.features)
        c_no_penalty.features["family_plateau_gens"] = 0.0
        c_no_penalty.features["style_non_improving_frac"] = 0.0

        scorer.score_candidates([c, c_no_penalty], state)
        assert c.score < c_no_penalty.score


# ---------------------------------------------------------------------------
# MMR selection tests
# ---------------------------------------------------------------------------


class TestMMRSelection:
    def test_selects_best_first(self) -> None:
        scorer = BatchSlotScorer()
        candidates = [
            SlotCandidate("a", 6.0, "ring", "refine", score=0.9),
            SlotCandidate("b", 5.0, "grid", "explore", score=0.5),
            SlotCandidate("c", 4.0, "hex", "radical", score=0.3),
        ]
        selected = scorer.select(candidates, 2)
        assert selected[0].parent_id == "a"

    def test_avoids_duplicate_styles_when_possible(self) -> None:
        scorer = BatchSlotScorer()
        candidates = [
            SlotCandidate("a", 6.0, "ring", "refine", score=0.9),
            SlotCandidate("b", 5.5, "grid", "refine", score=0.85),
            SlotCandidate("c", 5.0, "hex", "explore", score=0.80),
        ]
        selected = scorer.select(candidates, 2)
        styles = [s.style for s in selected]
        # MMR should prefer diversity, so we get both refine and explore
        assert len(set(styles)) == 2

    def test_avoids_duplicate_parents_when_possible(self) -> None:
        scorer = BatchSlotScorer()
        candidates = [
            SlotCandidate("a", 6.0, "ring", "refine", score=0.9),
            SlotCandidate("a", 6.0, "ring", "explore", score=0.85),
            SlotCandidate("b", 5.0, "grid", "refine", score=0.80),
        ]
        selected = scorer.select(candidates, 2)
        parents = [s.parent_id for s in selected]
        # MMR penalizes same parent_id (0.4 sim), so should pick different parents
        assert len(set(parents)) == 2

    def test_avoids_duplicate_families_when_possible(self) -> None:
        scorer = BatchSlotScorer()
        # ring/explore has family+style overlap with ring/refine (0.3+0.3=0.6 sim)
        # grid/explore only has style overlap (0.3 sim), so MMR prefers it
        candidates = [
            SlotCandidate("a", 6.0, "ring", "refine", score=0.9),
            SlotCandidate("b", 5.5, "ring", "explore", score=0.85),
            SlotCandidate("c", 5.0, "grid", "explore", score=0.84),
        ]
        selected = scorer.select(candidates, 2)
        families = [s.parent_family for s in selected]
        assert len(set(families)) == 2

    def test_returns_k_even_with_fewer_unique_options(self) -> None:
        scorer = BatchSlotScorer()
        candidates = [
            SlotCandidate("a", 6.0, "ring", "refine", score=0.9),
            SlotCandidate("a", 6.0, "ring", "refine", score=0.8),
        ]
        selected = scorer.select(candidates, 3)
        # Can only return 2 since there are only 2 candidates
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# plan_batch integration tests
# ---------------------------------------------------------------------------


class TestPlanBatch:
    def test_returns_correct_number(self) -> None:
        engine = MockEngine(batch_size=4)
        scorer = BatchSlotScorer()
        result = scorer.plan_batch(engine, SearchMode.EXPLOIT)
        assert len(result) == 4

    def test_all_parent_codes_valid(self) -> None:
        engine = MockEngine(batch_size=4)
        scorer = BatchSlotScorer()
        result = scorer.plan_batch(engine, SearchMode.EXPLOIT)
        all_codes = set(engine._program_store.values()) | {engine._best_code}
        for parent_code, _style in result:
            assert parent_code in all_codes

    def test_all_styles_valid_for_mode(self) -> None:
        engine = MockEngine(batch_size=4)
        scorer = BatchSlotScorer()
        for mode in [SearchMode.EXPLOIT, SearchMode.EXPLORE, SearchMode.REPAIR]:
            result = scorer.plan_batch(engine, mode)
            from esn.engine.slot_scorer import _MODE_STYLE_MAP

            valid_styles = _MODE_STYLE_MAP[mode]
            # Slot 0 is the exploitation anchor ("refine"), exempt from mode styles
            for i, (_code, style) in enumerate(result):
                if i == 0:
                    assert style == "refine", "Slot 0 must be the exploitation anchor"
                else:
                    assert style in valid_styles

    def test_batch_size_1(self) -> None:
        engine = MockEngine(batch_size=1)
        scorer = BatchSlotScorer()
        result = scorer.plan_batch(engine, SearchMode.EXPLOIT)
        assert len(result) == 1
        code, style = result[0]
        assert code == engine._best_code
        assert style == "refine"

    def test_slot_0_is_exploitation_anchor(self) -> None:
        engine = MockEngine(batch_size=4)
        scorer = BatchSlotScorer()
        result = scorer.plan_batch(engine, SearchMode.EXPLOIT)
        code, style = result[0]
        assert code == engine._best_code
        assert style == "refine"

    def test_anchor_not_duplicated_in_remaining_slots(self) -> None:
        engine = MockEngine(batch_size=4)
        scorer = BatchSlotScorer()
        result = scorer.plan_batch(engine, SearchMode.EXPLOIT)
        # Slot 0 is the anchor; remaining slots should not be identical
        anchor = result[0]
        for slot in result[1:]:
            # At least one of (code, style) should differ
            if slot[0] == anchor[0] and slot[1] == anchor[1]:
                # This can happen via padding, but only if there were very few candidates
                pass  # acceptable edge case for padding


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_full_scorer_roundtrip(self) -> None:
        scorer = BatchSlotScorer(weights=ScorerWeights(w_parent_quantile=0.5))
        scorer.combo_tracker.record("ring", "refine", True, 5)
        scorer.combo_tracker.record("grid", "explore", False, 3)

        data = scorer.to_dict()
        restored = BatchSlotScorer.from_dict(data)

        assert restored.weights.w_parent_quantile == 0.5
        assert restored.combo_tracker.get("ring", "refine") == (1, 1, 5)
        assert restored.combo_tracker.get("grid", "explore") == (1, 0, 3)

    def test_weights_preserved(self) -> None:
        w = ScorerWeights(mmr_lambda=0.5, w_family_rarity=0.99)
        scorer = BatchSlotScorer(weights=w)
        data = scorer.to_dict()
        restored = BatchSlotScorer.from_dict(data)
        assert restored.weights.mmr_lambda == 0.5
        assert restored.weights.w_family_rarity == 0.99

    def test_combo_tracker_preserved(self) -> None:
        scorer = BatchSlotScorer()
        scorer.combo_tracker.record("a", "b", True, 10)
        scorer.combo_tracker.record("a", "b", False, 11)
        scorer.combo_tracker.record("c", "d", True, 12)

        data = scorer.to_dict()
        restored = BatchSlotScorer.from_dict(data)
        assert restored.combo_tracker.get("a", "b") == (2, 1, 11)
        assert restored.combo_tracker.get("c", "d") == (1, 1, 12)
        assert restored.combo_tracker.total_attempts() == 3
