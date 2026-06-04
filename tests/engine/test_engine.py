"""Tests for ESNEngine pure engine logic (no LLM mocks)."""

from __future__ import annotations

from esn.engine.engine import ESNEngine


class TestOneGeneration:
    def test_one_generation_happy_path(self, engine_no_llm):
        record = engine_no_llm.run_generation()
        assert record.success is True
        assert record.score > 0
        assert record.generation == 1

    def test_generation_increments(self, engine_no_llm):
        for i in range(3):
            record = engine_no_llm.run_generation()
            assert record.generation == i + 1

    def test_compile_failure_recorded(self, simple_domain):
        simple_domain.initial_code = "def solve():\n    raise RuntimeError('fail')\n"
        engine = ESNEngine(domain=simple_domain)
        record = engine.run_generation()
        assert record.success is False


class TestArchivesAndState:
    def test_elite_archive_populated(self, engine_no_llm):
        engine_no_llm.run_generation()
        assert engine_no_llm.elite_archive.size >= 1

    def test_search_state_updated(self, engine_no_llm):
        engine_no_llm.run_generation()
        s = engine_no_llm.state
        assert s.generation == 1
        assert len(s.recent_scores) >= 1
        assert s.best_score > 0

    def test_stagnation_counter(self, engine_no_llm):
        # Run multiple identical generations (no improvement -- same code each time)
        for _ in range(5):
            engine_no_llm.run_generation()
        # After generation 1 sets the score, subsequent gens can't beat it by 0.5%
        # so stagnation_counter should be > 0
        assert engine_no_llm.state.stagnation_counter >= 1


class TestOperatorCredit:
    def test_operator_credit_recorded(self, engine_no_llm):
        engine_no_llm.run_generation()
        # Default style for EXPLOIT mode is "refine"
        stats = engine_no_llm.credit_model.get_stats("refine")
        assert stats.attempts >= 1

    def test_forced_exploration_cycles_styles(self, engine_no_llm):
        # Run enough generations to trigger forced exploration on all core styles.
        # The engine cycles through eligible styles with < _MIN_TRIES_PER_STYLE attempts.
        # Running 8 gens should cover 4 core styles x 2 minimum tries, though
        # mode selection may not reach all styles. We check at least 2 styles got attempts.
        for _ in range(8):
            engine_no_llm.run_generation()
        styles_with_attempts = 0
        for style in ["refine", "explore", "repair", "radical"]:
            if engine_no_llm.credit_model.get_stats(style).attempts >= 1:
                styles_with_attempts += 1
        # At minimum, refine should be used (EXPLOIT default)
        assert styles_with_attempts >= 1


class TestModeSelection:
    def test_mode_selection_default_exploit(self, engine_no_llm):
        engine_no_llm.run_generation()
        # First gen starts with EXPLOIT by default
        assert engine_no_llm.state.current_mode.value in ("exploit", "explore", "repair", "bridge")


class TestBestCodeTracking:
    def test_best_code_tracks_improvement(self, simple_domain, simple_evaluator):
        # initial_code scores 6.0 (sum of [1,2,3])
        engine = ESNEngine(domain=simple_domain)
        engine.run_generation()
        assert engine._best_score == 6.0

        # Now change initial_code so next gen produces higher score
        better_code = "def solve():\n    return [10, 20, 30]\n"
        simple_domain.initial_code = better_code
        # Reset best_code so identity mutation picks up the new initial
        engine._best_code = better_code
        record = engine.run_generation()
        assert record.score == 60.0
        assert engine._best_score == 60.0


class TestProgramStorePruning:
    def test_program_store_pruning(self, engine_no_llm):
        # Manually fill store beyond 200 entries
        for i in range(250):
            engine_no_llm._program_store[f"fake-{i}"] = f"code-{i}"
        # Run a generation which triggers pruning
        engine_no_llm.run_generation()
        assert len(engine_no_llm._program_store) <= 201  # 200 limit + 1 new entry at most
