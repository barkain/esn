# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Real spectral + epistemic novelty for ESN core engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from esn.core.epistemic import compute_epistemic_novelty, normalize_epistemic
from esn.core.spectral_models import ESNConfig, HypothesisRecord, SpectralState
from esn.core.scorer import compute_unified_novelty
from esn.core.spectral import compute_gram_schmidt_residual, run_spectral_pipeline
from esn.core.spectral_calibration import (
    DEFAULT_ALIGNMENT_GATE,
    SpectralReport,
    actionable_spikes,
    analyze_spectrum,
    report_to_dict,
)


class NoveltyComputer:
    """Computes real epistemic and spectral novelty for core engine.

    Wraps v1's spectral pipeline, epistemic scoring, and unified novelty.
    Call end_of_generation() once per generation after all hypothesis updates.
    Call compute() per candidate to get novelty scores.
    """

    def __init__(
        self,
        knowledge: Any,  # KnowledgeIntegration
        config: ESNConfig | None = None,
        observation_providers: list[Any] | None = None,  # SpectralObservationProvider list
        seed: int | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._config = config or ESNConfig()
        self._spectral_state: SpectralState | None = None
        self._spectral_report: SpectralReport | None = None
        self._spike_count_history: list[int] = []
        self._max_epistemic: float = 0.0
        self._observation_providers: list[Any] = observation_providers or []
        self._max_spectral: float = 0.0
        # Isolated RNG for spectral empirical-null sampling.
        # When ``seed`` is provided, the empirical-null draws are deterministic
        # and independent of the numpy global RNG state shifted by torch /
        # sentence-transformers init (Phase 0.1 RNG determinism fix). When no
        # seed is provided, fall back to a fresh entropy-seeded Generator so
        # unseeded callers still get independent draws across runs/instances.
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Spectral compression
        from esn.core.spectral_compression import SpectralCompressor

        self._compressor = SpectralCompressor(target_dim=self._config.spectral_dim)

    @property
    def spectral_state(self) -> SpectralState | None:
        return self._spectral_state

    @property
    def spectral_report(self) -> SpectralReport | None:
        """BBP-grounded calibration report for the current spectral state.

        Additive to ``spectral_state``. ``None`` until ``end_of_generation``
        produces a non-empty spectral state.
        """
        return self._spectral_report

    def set_observation_providers(self, providers: list) -> None:
        """Set observation providers after init."""
        self._observation_providers = providers

    def end_of_generation(self) -> None:
        """Run spectral pipeline on current knowledge bank.

        Called once per generation AFTER all hypothesis updates.
        Uses PCA compression and observation expansion.
        """
        from esn.core.spectral_compression import compress_records, expand_observations

        active = self._knowledge.get_active_hypothesis_records()

        # Expand with additional observations
        all_records = expand_observations(
            active,
            self._observation_providers,
            embedder=getattr(self._knowledge, "_embedder", None),
            embedding_dim=getattr(self._knowledge, "_embedding_dim", 1024),
        )

        # Compress to spectral working dimension
        compressed = compress_records(all_records, self._compressor)

        new_state = run_spectral_pipeline(
            compressed,
            prev_state=self._spectral_state,
            threshold_mode=self._config.spectral_threshold_mode,
            rng=self._rng,
        )
        if new_state is not None:
            # Annotate state with dimension info
            new_state.spectral_dim = self._config.spectral_dim
            new_state.observation_count = len(all_records)
            self._spectral_state = new_state
            self._spike_count_history.append(new_state.num_spikes)

            # Phase 1 (additive): build a BBP-grounded SpectralReport from the
            # same eigenvalues. This sits ALONGSIDE the existing SpectralState
            # fields — it does not replace num_spikes / mp_threshold logic.
            # The controller will eventually consume per-spike alignment²
            # instead of raw spike counts.
            try:
                self._spectral_report = analyze_spectrum(
                    new_state.eigenvalues,
                    sigma2=float(new_state.sigma_sq),
                    gamma=float(new_state.gamma_t),
                    n_obs=int(new_state.observation_count),
                    alignment_gate=DEFAULT_ALIGNMENT_GATE,
                )
            except Exception:  # noqa: BLE001 - report is purely additive
                self._spectral_report = None
        else:
            self._spike_count_history.append(0)
            self._spectral_report = None

    def compute(
        self,
        relevant_data: list[dict],  # [{confidence, delta}] from process_analysis
        new_count: int,
        engaged_hypotheses: list[HypothesisRecord],
        actual_score: float,
        prediction_surprise: bool = False,
    ) -> tuple[float, float, float]:
        """Compute (epistemic_novelty, spectral_novelty, unified_novelty)."""

        # 1. Epistemic novelty (Eq 7)
        ep_raw = compute_epistemic_novelty(
            relevant_hypotheses=relevant_data,
            new_hypothesis_count=new_count,
            prediction_surprise=prediction_surprise,
            alpha=self._config.alpha,
            beta=self._config.beta,
            actual_score=actual_score,
            failure_threshold=self._config.failure_threshold,
            failure_discount=self._config.failure_discount,
        )
        self._max_epistemic = max(self._max_epistemic, ep_raw)
        ep_norm = normalize_epistemic(ep_raw, self._max_epistemic)

        # 2. Spectral novelty (Gram-Schmidt residual, Eq 12)
        sp_score: float | None = None
        if self._spectral_state is not None and engaged_hypotheses:
            # Compress engaged hypotheses for spectral computation
            compressed_engaged = []
            for h in engaged_hypotheses:
                comp_emb = self._compressor.transform(h.embedding)
                compressed_engaged.append(h.model_copy(update={"embedding": comp_emb}))

            sp_score = compute_gram_schmidt_residual(
                compressed_engaged,
                self._spectral_state.V_k,
                self._spectral_state.mean_row,
            )

        if sp_score is not None:
            self._max_spectral = max(self._max_spectral, sp_score)

        # If no spikes exist (legacy empirical detector AND BBP actionable both
        # empty), spectral novelty is uninformative — return None so
        # compute_unified_novelty falls back to pure epistemic. Phase 1 follow-up:
        # honor BBP actionable spikes as a valid structure signal too.
        bbp_actionable = 0
        if self._spectral_report is not None and not self._spectral_report.undersampled:
            bbp_actionable = len(actionable_spikes(self._spectral_report))
        legacy_spikes = self._spectral_state.num_spikes if self._spectral_state else 0
        effective_spikes = max(legacy_spikes, bbp_actionable)
        if self._spectral_state is None or effective_spikes == 0:
            sp_score = None

        # 3. Unified novelty (Eq 13-14 with signal-quality gate)
        current_spikes = effective_spikes
        unified, _ = compute_unified_novelty(
            ep_norm,
            sp_score,
            self._spectral_state,
            self._config.tau,
            n_persistent_spikes=current_spikes,
            spike_persistence_gens=self._spike_persistence,
            min_spike_persistence=self._config.min_spike_persistence,
        )

        return ep_norm, sp_score if sp_score is not None else 0.0, unified

    def select_cluster_representatives(
        self,
        active_records: list[HypothesisRecord],
        per_cluster: int = 2,
        limit: int = 10,
    ) -> list[HypothesisRecord] | None:
        """Phase 3.10: pick per-spike representative hypotheses for prompts.

        Returns a deduplicated list of hypotheses that are the strongest
        projectors onto the actionable spike directions (alignment² above the
        Phase 1 BBP gate). Returns ``None`` when no spectral state / no
        actionable spikes are available, so callers can fall back to the
        confidence-sorted behavior.

        ``per_cluster`` caps representatives per spike; ``limit`` caps the
        total result length after deduplication.
        """
        state = self._spectral_state
        report = self._spectral_report
        if state is None or state.V_k is None or state.mean_row is None:
            return None
        if report is None:
            return None
        # Phase 1 follow-up guardrail: in the undersampled regime (n_obs<30 or
        # gamma>0.9) BBP alignments are too noisy to steer prompt composition,
        # so fall back to the legacy confidence-sort path.
        if report.undersampled:
            return None
        actionable = [s for s in report.spikes if s.above_gate]
        if not actionable:
            return None
        if not active_records:
            return []

        # Compress each record's embedding to the spectral working dim,
        # center using the SAME mean_row that defines this spectral state,
        # then project onto V_k. Any record whose compressed dim doesn't
        # match V_k's rows (e.g. compressor wasn't fit yet) is skipped.
        V_k = np.asarray(state.V_k)
        mean_row = np.asarray(state.mean_row)
        projections: list[np.ndarray] = []
        usable_records: list[HypothesisRecord] = []
        for rec in active_records:
            try:
                comp = self._compressor.transform(rec.embedding)
            except Exception:  # noqa: BLE001
                continue
            comp = np.asarray(comp).ravel()
            if comp.shape[0] != mean_row.shape[0]:
                continue
            centered = comp - mean_row
            proj = centered @ V_k  # (k,)
            projections.append(proj)
            usable_records.append(rec)

        if not usable_records:
            return None

        proj_matrix = np.vstack(projections)  # (H, k)
        selected_ids: list[str] = []
        selected: list[HypothesisRecord] = []
        for spike in actionable:
            col = spike.rank
            if col >= proj_matrix.shape[1]:
                continue
            order = np.argsort(-np.abs(proj_matrix[:, col]))
            picked = 0
            for idx in order:
                rec = usable_records[int(idx)]
                if rec.id in selected_ids:
                    continue
                selected.append(rec)
                selected_ids.append(rec.id)
                picked += 1
                if picked >= per_cluster or len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break
        return selected

    @property
    def _spike_persistence(self) -> int:
        """Count consecutive recent generations with at least one spike."""
        count = 0
        for n in reversed(self._spike_count_history):
            if n > 0:
                count += 1
            else:
                break
        return count

    @property
    def max_spikes_seen(self) -> int:
        """Maximum spike count observed in any single generation."""
        return max(self._spike_count_history) if self._spike_count_history else 0

    @property
    def gens_with_spikes(self) -> int:
        """Number of generations that had at least one spike."""
        return sum(1 for s in self._spike_count_history if s > 0)

    @property
    def max_spectral_novelty(self) -> float:
        """Maximum spectral novelty score observed across all candidates."""
        return self._max_spectral

    @property
    def spectral_guidance(self) -> dict[str, Any]:
        """Spectral guidance dict for MutationContext."""
        if self._spectral_state is None:
            return {}
        bbp: dict[str, Any] = {}
        if self._spectral_report is not None:
            bbp = {
                "bbp": report_to_dict(self._spectral_report),
                "actionable_spike_count": len(actionable_spikes(self._spectral_report)),
                "dominant_alignment_sq": self._spectral_report.dominant_alignment,
                "effective_spike_count": self._spectral_report.effective_spike_count,
                "spectral_undersampled": self._spectral_report.undersampled,
            }
        return {
            **bbp,
            "mutation_guidance": self._spectral_state.mutation_guidance,
            "num_spikes": self._spectral_state.num_spikes,
            "erank": self._spectral_state.erank,
            "S1": self._spectral_state.S1,
            "S2": self._spectral_state.S2,
            "gamma": self._spectral_state.gamma_t,
            "spectral_dim": self._spectral_state.spectral_dim,
            "observation_count": self._spectral_state.observation_count,
            "mp_threshold": self._spectral_state.mp_threshold,
            "empirical_threshold": self._spectral_state.empirical_threshold or 0.0,
            "max_eigenvalue": self._spectral_state.max_eigenvalue,
            "max_eigen_mp_ratio": self._spectral_state.max_eigen_mp_ratio,
            "empirical_mp_ratio": self._spectral_state.empirical_mp_ratio,
            "max_spikes_seen": self.max_spikes_seen,
            "gens_with_spikes": self.gens_with_spikes,
        }
