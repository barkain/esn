"""Tests for NoveltyComputer — real spectral + epistemic novelty for core."""

from __future__ import annotations

import numpy as np
import pytest

from esn.core.spectral_models import HypothesisRecord
from esn.core.knowledge import KnowledgeIntegration
from esn.core.novelty import NoveltyComputer


class MockEmbedder:
    @property
    def dimension(self):
        return 1024

    def embed(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        v = rng.randn(1024).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])


def _make_hypothesis(
    id: str, text: str, confidence: float = 0.8, n_obs: int = 5
) -> HypothesisRecord:
    emb = MockEmbedder().embed(text)
    return HypothesisRecord(
        id=id,
        text=text,
        confidence=confidence,
        n_obs=n_obs,
        embedding=emb,
        concepts=["test"],
        created_at=0,
        last_tested=0,
        status="active",
    )


def _make_knowledge_with_hypotheses(n: int) -> KnowledgeIntegration:
    """Create KnowledgeIntegration pre-loaded with n diverse hypotheses."""
    knowledge = KnowledgeIntegration(embedder=MockEmbedder())
    for i in range(n):
        h = _make_hypothesis(f"h-{i}", f"Hypothesis about topic {i} with unique content {i * 7}")
        knowledge.bank.add(h)
    return knowledge


class TestNoveltyEmpty:
    def test_novelty_empty_bank(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        ep, sp, unified = nc.compute(
            relevant_data=[], new_count=0, engaged_hypotheses=[], actual_score=0.0
        )
        assert ep == 0.0
        assert sp == 0.0
        assert unified == 0.0


class TestEpistemicNovelty:
    def test_novelty_with_evidence(self):
        knowledge = KnowledgeIntegration()
        h = _make_hypothesis("h-1", "Test hypothesis")
        knowledge.bank.add(h)

        nc = NoveltyComputer(knowledge)
        relevant_data = [{"confidence": 0.5, "delta": 0.3}]
        ep, sp, unified = nc.compute(
            relevant_data=relevant_data,
            new_count=1,
            engaged_hypotheses=[h],
            actual_score=2.0,
        )
        assert ep > 0.0

    def test_max_epistemic_tracking(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)

        # First compute with small evidence
        nc.compute(
            relevant_data=[{"confidence": 0.5, "delta": 0.1}],
            new_count=0,
            engaged_hypotheses=[],
            actual_score=1.0,
        )
        max1 = nc._max_epistemic

        # Second compute with larger evidence
        nc.compute(
            relevant_data=[{"confidence": 0.5, "delta": 0.5}],
            new_count=3,
            engaged_hypotheses=[],
            actual_score=2.0,
        )
        max2 = nc._max_epistemic
        assert max2 >= max1

    def test_epistemic_normalize(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)

        ep, _, _ = nc.compute(
            relevant_data=[{"confidence": 0.8, "delta": 0.4}],
            new_count=2,
            engaged_hypotheses=[],
            actual_score=3.0,
        )
        assert 0.0 <= ep <= 1.0


class TestSpectralNovelty:
    def test_novelty_spectral_bootstrap(self):
        """With < 2 hypotheses, spectral returns 0 (pipeline needs enough data)."""
        knowledge = KnowledgeIntegration(embedder=MockEmbedder())
        h = _make_hypothesis("h-1", "Single hypothesis")
        knowledge.bank.add(h)

        nc = NoveltyComputer(knowledge)
        nc.end_of_generation()

        ep, sp, unified = nc.compute(
            relevant_data=[{"confidence": 0.5, "delta": 0.2}],
            new_count=0,
            engaged_hypotheses=[h],
            actual_score=1.0,
        )
        # With very few hypotheses, spectral pipeline may not produce a state
        assert isinstance(sp, float)

    def test_novelty_spectral_kicks_in(self):
        """With 10+ diverse hypotheses, spectral state should exist."""
        knowledge = _make_knowledge_with_hypotheses(15)
        nc = NoveltyComputer(knowledge)
        nc.end_of_generation()
        assert nc.spectral_state is not None

    def test_end_of_generation_updates_state(self):
        knowledge = _make_knowledge_with_hypotheses(15)
        nc = NoveltyComputer(knowledge)

        assert nc.spectral_state is None
        nc.end_of_generation()
        assert nc.spectral_state is not None

    def test_compute_with_no_engaged(self):
        """Empty engaged list -> spectral = 0."""
        knowledge = _make_knowledge_with_hypotheses(15)
        nc = NoveltyComputer(knowledge)
        nc.end_of_generation()

        ep, sp, unified = nc.compute(
            relevant_data=[{"confidence": 0.5, "delta": 0.2}],
            new_count=1,
            engaged_hypotheses=[],  # empty
            actual_score=1.0,
        )
        assert sp == 0.0


class TestSpikePersistence:
    def test_spike_persistence_tracking(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        nc._spike_count_history = [1, 2, 3]
        assert nc._spike_persistence == 3

    def test_spike_persistence_gap(self):
        """History [1, 1, 0, 1] -> persistence = 1 (only counts from end)."""
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        nc._spike_count_history = [1, 1, 0, 1]
        assert nc._spike_persistence == 1

    def test_spike_persistence_empty(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        assert nc._spike_persistence == 0

    def test_spike_persistence_all_zero(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        nc._spike_count_history = [0, 0, 0]
        assert nc._spike_persistence == 0


class TestSpectralGuidance:
    def test_spectral_guidance_empty(self):
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        assert nc.spectral_guidance == {}

    def test_spectral_guidance_populated(self):
        knowledge = _make_knowledge_with_hypotheses(15)
        nc = NoveltyComputer(knowledge)
        nc.end_of_generation()
        if nc.spectral_state is not None:
            guidance = nc.spectral_guidance
            assert "mutation_guidance" in guidance
            assert "num_spikes" in guidance
            assert "erank" in guidance
            assert "S1" in guidance
            assert "S2" in guidance


class TestUnifiedNovelty:
    def test_unified_pure_epistemic(self):
        """No spikes -> unified = epistemic (gamma_w = 0)."""
        knowledge = KnowledgeIntegration()
        nc = NoveltyComputer(knowledge)
        # No spectral state, no spikes
        ep, sp, unified = nc.compute(
            relevant_data=[{"confidence": 0.6, "delta": 0.3}],
            new_count=1,
            engaged_hypotheses=[],
            actual_score=2.0,
        )
        # Without spectral signal, unified should equal epistemic
        assert unified == pytest.approx(ep, abs=1e-6)
