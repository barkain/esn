# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Unified novelty scoring (Section 5.2, Eq 13-14)."""

from typing import Optional
from .spectral_models import SpectralState
from .utils import sigmoid


def compute_gamma_weight(
    erank: float,
    tau: float = 5.0,
    n_persistent_spikes: int = 0,
    spike_persistence_gens: int = 0,
    min_persistence: int = 3,
) -> float:
    """Adaptive mixing weight (Eq 14) with signal-quality gate.

    Spectral weight is near zero unless:
    - At least one spike exists above the threshold (n_persistent_spikes > 0)
    - Spikes have persisted for min_persistence consecutive generations

    When those conditions are met, use the paper's sigmoid formula:
    gamma_w(t) = sigmoid(-erank(Sigma_t) / tau)

    When erank is low (immature bank) -> gamma_w high -> more spectral weight.
    When erank is high (mature bank)   -> gamma_w low  -> more epistemic weight.
    Default tau = 5 (crossover at ~5 independent clusters).
    """
    if n_persistent_spikes == 0 or spike_persistence_gens < min_persistence:
        return 0.0

    return sigmoid(-erank / tau)


def compute_unified_novelty(
    epistemic_score_normalized: float,
    spectral_score: Optional[float],
    spectral_state: Optional[SpectralState],
    tau: float = 5.0,
    n_persistent_spikes: int = 0,
    spike_persistence_gens: int = 0,
    min_spike_persistence: int = 3,
) -> tuple[float, float]:
    """Compute the unified novelty score N(x,t) per Eq 13.

    N(x,t) = gamma_w * N_sp_tilde(x,t) + (1 - gamma_w) * N_ep_tilde(x)

    Args:
        epistemic_score_normalized: N_ep_tilde in [0, 1].
        spectral_score: N_sp_tilde(x,t) from Gram-Schmidt residual, or None.
        spectral_state: Current SpectralState (for erank), or None.
        tau: Temperature for adaptive mixing.
        n_persistent_spikes: Number of spikes in the most recent generation.
        spike_persistence_gens: Consecutive generations with at least one spike.
        min_spike_persistence: Minimum consecutive gens with spikes to activate.

    Returns:
        (unified_score, gamma_weight)
    """
    if spectral_score is None or spectral_state is None:
        # No spectral data available; unified = epistemic only
        return epistemic_score_normalized, 0.0

    gamma_w = compute_gamma_weight(
        spectral_state.erank,
        tau,
        n_persistent_spikes=n_persistent_spikes,
        spike_persistence_gens=spike_persistence_gens,
        min_persistence=min_spike_persistence,
    )
    unified = gamma_w * spectral_score + (1.0 - gamma_w) * epistemic_score_normalized

    return unified, gamma_w
