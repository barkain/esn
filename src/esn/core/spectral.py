# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Spectral novelty: random-matrix-theory analysis of the knowledge bank.

Implements Sections 4 and 5.1 of the ESN paper:
- Eq 3: Knowledge Structure Matrix K_t
- Eq 4: Centering and covariance Sigma_t
- Eq 8: Marchenko-Pastur law (lambda_+, lambda_-)
- Eq 9: Tracy-Widom finite-sample correction
- Eq 10-11: Four spectral signals S1-S4
- Eq 12: Per-solution Gram-Schmidt residual
"""

import logging
import numpy as np
from typing import Optional
from .spectral_models import HypothesisRecord, SpectralState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Structure Matrix (Eq 3)
# ---------------------------------------------------------------------------


def build_knowledge_matrix(knowledge_bank: list[HypothesisRecord]) -> np.ndarray:
    """Build K_t where row i = w_i * e_i, with certainty weighting.

    w_i = 2 * |c_i - 0.5| — measures knowledge certainty, not belief direction.
    Confirmed (c≈0.9) and refuted (c≈0.1) hypotheses both contribute strongly.
    Untested hypotheses (c=0.5) contribute nothing.

    Returns H_t x d matrix. Empty bank returns shape (0, 0).
    """
    if not knowledge_bank:
        return np.array([]).reshape(0, 0)

    embeddings = np.array([h.embedding for h in knowledge_bank])
    confidences = np.array([h.confidence for h in knowledge_bank])
    certainty = 2 * np.abs(confidences - 0.5)
    return certainty[:, np.newaxis] * embeddings


# ---------------------------------------------------------------------------
# Centering and covariance (Eq 4)
# ---------------------------------------------------------------------------


def center_matrix(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center K_t by subtracting the mean row (Eq 4).

    K_tilde = K - (1/H_t) * 1 * 1^T * K

    Returns (K_tilde, mean_row).
    """
    mean_row = K.mean(axis=0)  # shape (d,)
    K_tilde = K - mean_row[np.newaxis, :]
    return K_tilde, mean_row


def compute_covariance_eigenvalues(
    K_tilde: np.ndarray, H_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenvalues of Sigma_t = (1/H_t) * K_tilde^T K_tilde via SVD of K_tilde.

    The SVD gives K_tilde = U diag(sigma) V^T, so
    eigenvalues of Sigma_t = sigma^2 / H_t.

    Returns (eigenvalues_descending, singular_values, V) where V contains
    the right singular vectors (columns are vectors).
    """
    U, sigma, Vt = np.linalg.svd(K_tilde, full_matrices=False)
    eigenvalues = sigma**2 / H_t
    # Sort descending (SVD already returns sigma descending)
    return eigenvalues, sigma, Vt.T  # V = Vt.T so columns are right singular vectors


# ---------------------------------------------------------------------------
# Marchenko-Pastur parameters (Eq 8)
# ---------------------------------------------------------------------------


def compute_mp_parameters(d: int, H_t: int, sigma_sq: float) -> tuple[float, float, float]:
    """Compute MP law parameters (Eq 8).

    gamma_t = d / H_t
    lambda_pm = sigma^2 * (1 +/- sqrt(gamma_t))^2

    Returns (gamma_t, lambda_plus, lambda_minus).
    """
    gamma_t = d / H_t
    sqrt_gamma = np.sqrt(gamma_t)
    lambda_plus = sigma_sq * (1 + sqrt_gamma) ** 2
    lambda_minus = sigma_sq * max(0.0, (1 - sqrt_gamma) ** 2)
    return gamma_t, lambda_plus, lambda_minus


def estimate_noise_variance(eigenvalues: np.ndarray, gamma_t: float) -> float:
    """Estimate sigma^2 (per-dimension noise variance) from the bulk eigenvalues.

    The Marchenko-Pastur law parametrises the bulk via sigma^2, which is
    the average *per-dimension* variance of the data matrix entries.
    Equivalently, sigma^2 = trace(Sigma_t) / d.

    When gamma_t = d / H_t, the SVD of the (H_t x d) centered matrix
    yields min(H_t, d) eigenvalues.  Their sum equals trace(Sigma_t).
    Dividing by d gives the correct per-dimension estimate.

    When gamma_t <= 1 (more samples than dimensions), len(eigenvalues) == d,
    so trace / d == mean(eigenvalues).  When gamma_t > 1 (high-dimensional
    regime, d > H_t), len(eigenvalues) == H_t < d, and we must account for
    the (d - H_t) implicit zero eigenvalues by dividing by d, not H_t.
    """
    if len(eigenvalues) == 0:
        return 1.0

    trace = float(np.sum(eigenvalues))
    H_t = len(eigenvalues)
    d = int(round(gamma_t * H_t))  # gamma_t = d / H_t
    # Guard against degenerate cases
    if d < 1:
        d = H_t
    return trace / d


# ---------------------------------------------------------------------------
# Tracy-Widom correction (Eq 9)
# ---------------------------------------------------------------------------


def compute_tw_correction(
    lambda_plus: float,
    H_t: int,
    tw_quantile: float = 0.979,
) -> float:
    """Tracy-Widom finite-sample correction (Eq 9).

    lambda_plus_corrected = lambda_plus + (lambda_plus^{2/3} / H_t) * F_TW1^{-1}(1-p)

    With p = 0.05, F_TW1^{-1}(0.95) ~ 0.979.
    """
    correction = (lambda_plus ** (2.0 / 3.0) / H_t) * tw_quantile
    return lambda_plus + correction


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------


def detect_spikes(eigenvalues: np.ndarray, lambda_plus_corrected: float) -> np.ndarray:
    """Return indices of eigenvalues exceeding the TW-corrected MP edge.

    J_t = {j : lambda_j > lambda_plus_corrected}
    """
    return np.where(eigenvalues > lambda_plus_corrected)[0]


# ---------------------------------------------------------------------------
# Effective rank (Eq 11) — computed on eigenvalues of Sigma_t
# ---------------------------------------------------------------------------


def compute_effective_rank(eigenvalues: np.ndarray) -> float:
    """Effective rank via Shannon entropy of normalized eigenvalues (Eq 11).

    erank(Sigma_t) = exp(-SUM lambda_bar_i * ln(lambda_bar_i))
    where lambda_bar_i = lambda_i / SUM_j lambda_j.
    """
    pos = eigenvalues[eigenvalues > 1e-12]
    if len(pos) == 0:
        return 0.0
    lam_bar = pos / pos.sum()
    entropy = -np.sum(lam_bar * np.log(lam_bar))
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Four spectral signals (Section 4.3)
# ---------------------------------------------------------------------------


def compute_S1(num_spikes_new: int, num_spikes_old: int) -> float:
    """S1: Spike emergence (Eq 10).

    S1(t) = |spikes at t+1| - |spikes at t|
    """
    return float(num_spikes_new - num_spikes_old)


def compute_S2(erank_new: float, erank_old: float) -> float:
    """S2: Effective rank change (Eq 11).

    S2(t) = erank(Sigma_{t+1}) - erank(Sigma_t)
    """
    return erank_new - erank_old


def compute_S3(eigenvalues: np.ndarray, lambda_plus_corrected: float) -> Optional[float]:
    """S3: Power-law exponent alpha_t via MLE (Clauset et al.).

    Fit lambda ~ i^{-alpha} for eigenvalues above the MP edge.
    Returns None if fewer than 3 spikes.
    """
    spike_vals = eigenvalues[eigenvalues > lambda_plus_corrected]
    if len(spike_vals) < 3:
        return None

    # MLE for power-law on ranked spike eigenvalues: lambda_j ~ j^{-alpha}
    # Take log: log(lambda_j) ~ -alpha * log(j)
    # MLE: alpha = n / SUM ln(lambda_j / lambda_min)
    n = len(spike_vals)
    spike_sorted = np.sort(spike_vals)  # ascending
    x_min = spike_sorted[0]
    if x_min <= 0:
        return None
    alpha = n / np.sum(np.log(spike_sorted / x_min))
    if not np.isfinite(alpha) or alpha <= 0:
        return None
    return float(alpha)


def compute_S4(
    eigenvalues_new: np.ndarray,
    eigenvalues_old: np.ndarray,
    gamma_new: float,
    gamma_old: float,
    sigma_sq_new: float,
    sigma_sq_old: float,
) -> float:
    """S4: Spectral divergence from null (Section 4.3).

    S4(t) = W1(F_hat_{Sigma_{t+1}}, F_MP) - W1(F_hat_{Sigma_t}, F_MP)

    Uses Wasserstein-1 distance between the empirical eigenvalue CDF and
    the theoretical MP CDF, computed numerically on a fine grid.
    """
    w1_new = _wasserstein1_vs_mp(eigenvalues_new, gamma_new, sigma_sq_new)
    w1_old = _wasserstein1_vs_mp(eigenvalues_old, gamma_old, sigma_sq_old)
    return w1_new - w1_old


def _wasserstein1_vs_mp(
    eigenvalues: np.ndarray,
    gamma: float,
    sigma_sq: float,
    n_grid: int = 1000,
) -> float:
    """Wasserstein-1 distance between empirical eigenvalue CDF and MP CDF.

    W1 = integral |F_hat(x) - F_MP(x)| dx, computed on a uniform grid.
    """
    if len(eigenvalues) == 0 or gamma <= 0 or sigma_sq <= 0:
        return 0.0

    sqrt_gamma = np.sqrt(gamma)
    lam_minus = sigma_sq * max(0.0, (1 - sqrt_gamma) ** 2)
    lam_plus = sigma_sq * (1 + sqrt_gamma) ** 2

    # Grid from 0 to max(eigenvalues, lambda_plus) * 1.1
    x_max = max(float(np.max(eigenvalues)), lam_plus) * 1.1
    x_grid = np.linspace(0, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    # Empirical CDF
    sorted_eigs = np.sort(eigenvalues)
    empirical_cdf = np.searchsorted(sorted_eigs, x_grid, side="right") / len(eigenvalues)

    # Theoretical MP CDF (numerical integration of the density)
    mp_cdf = np.zeros(n_grid)
    for i, x in enumerate(x_grid):
        if x <= lam_minus:
            # Point mass at 0 when gamma > 1
            mp_cdf[i] = max(0.0, 1.0 - 1.0 / gamma) if gamma > 1 else 0.0
        elif x >= lam_plus:
            mp_cdf[i] = 1.0
        else:
            # Integrate density from lam_minus to x
            point_mass = max(0.0, 1.0 - 1.0 / gamma) if gamma > 1 else 0.0
            sub_grid = np.linspace(lam_minus + 1e-10, x, 200)
            densities = _mp_density(sub_grid, gamma, sigma_sq)
            mp_cdf[i] = point_mass + np.trapezoid(densities, sub_grid)

    # W1 = integral |F - G| dx
    return float(np.sum(np.abs(empirical_cdf - mp_cdf)) * dx)


def _mp_density(x: np.ndarray, gamma: float, sigma_sq: float) -> np.ndarray:
    """Marchenko-Pastur density (Eq 8).

    rho_MP(lambda) = sqrt((lambda_+ - lambda)(lambda - lambda_-)) / (2*pi*gamma*sigma^2*lambda)
    """
    sqrt_gamma = np.sqrt(gamma)
    lam_minus = sigma_sq * max(0.0, (1 - sqrt_gamma) ** 2)
    lam_plus = sigma_sq * (1 + sqrt_gamma) ** 2

    result = np.zeros_like(x)
    mask = (x > lam_minus) & (x < lam_plus) & (x > 0)
    if np.any(mask):
        xm = x[mask]
        numerator = np.sqrt((lam_plus - xm) * (xm - lam_minus))
        denominator = 2.0 * np.pi * gamma * sigma_sq * xm
        result[mask] = numerator / denominator
    return result


# ---------------------------------------------------------------------------
# Per-solution Gram-Schmidt residual (Section 5.1, Eq 12)
# ---------------------------------------------------------------------------


def compute_gram_schmidt_residual(
    relevant_hypotheses: list[HypothesisRecord],
    V_k: Optional[np.ndarray],
    mean_row: Optional[np.ndarray],
) -> float:
    """Per-solution spectral novelty via Gram-Schmidt residual (Eq 12).

    1. e_x = weighted average of e_i with weights c_i
    2. e_tilde_x = e_x - k_bar      (center against stored mean)
    3. N_sp(x,t) = ||r||^2 / ||e_tilde_x||^2  where r = e_tilde - V_k V_k^T e_tilde

    V_k contains top-k right singular vectors of K_tilde as columns.

    Returns 1.0 (fully novel) when V_k, mean_row, or hypotheses are missing.
    """
    if len(relevant_hypotheses) == 0:
        return 1.0

    if V_k is None or mean_row is None:
        return 1.0

    # Step 1: certainty-weighted average of embeddings
    embeddings = np.array([h.embedding for h in relevant_hypotheses])
    confidences = np.array([h.confidence for h in relevant_hypotheses])
    certainty = 2 * np.abs(confidences - 0.5)
    # Fall back to uniform weights if all certainty is zero (all untested)
    if certainty.sum() < 1e-10:
        e_x = embeddings.mean(axis=0)
    else:
        e_x = np.average(embeddings, axis=0, weights=certainty)  # (d,)

    # Step 2: center against stored mean of K_t
    e_tilde = e_x - mean_row

    # Step 3: Gram-Schmidt residual fraction
    e_norm_sq = np.dot(e_tilde, e_tilde)
    if e_norm_sq < 1e-12:
        return 1.0

    # Project onto known subspace and compute residual
    projection = V_k @ (V_k.T @ e_tilde)
    residual = e_tilde - projection

    return float(np.dot(residual, residual) / e_norm_sq)


# ---------------------------------------------------------------------------
# Unexplored direction identification (spectral-guided diversity)
# ---------------------------------------------------------------------------


def identify_unexplored_directions(
    spectral_state: "SpectralState",
    hypotheses: list[HypothesisRecord],
    projection_threshold: float = 0.3,
) -> list[str]:
    """Find concept directions NOT covered by the top-k spike eigenvectors.

    1. Get the top-k eigenvectors (the 'known' subspace)
    2. For each hypothesis, compute how much it projects onto the known subspace
    3. Hypotheses with LOW projection are in the unexplored subspace
    4. Extract their concept tags — these are the unexplored concepts

    Args:
        spectral_state: Current spectral state with V_k
        hypotheses: Active hypotheses to analyze
        projection_threshold: Fraction below which a hypothesis is considered unexplored

    Returns:
        List of concept tags representing unexplored directions
    """
    if spectral_state is None or spectral_state.V_k is None:
        return []

    V_k = spectral_state.V_k  # top-k right singular vectors (columns)

    explored_concepts: set[str] = set()
    unexplored_concepts: set[str] = set()

    for h in hypotheses:
        if h.embedding is None or len(h.embedding) == 0:
            continue
        # Normalize the hypothesis embedding
        e = h.embedding / (np.linalg.norm(h.embedding) + 1e-10)
        # Project onto known subspace
        projection = V_k.T @ e  # shape (k,)
        projection_magnitude = np.linalg.norm(projection)

        # If most of the hypothesis is OUTSIDE the known subspace, it's unexplored
        if projection_magnitude < projection_threshold:
            for tag in h.concepts or []:
                unexplored_concepts.add(tag)
        else:
            for tag in h.concepts or []:
                explored_concepts.add(tag)

    # Return concepts that are unexplored but not also explored
    return sorted(unexplored_concepts - explored_concepts)


# ---------------------------------------------------------------------------
# Empirical null threshold (Section 1 of improvements)
# ---------------------------------------------------------------------------


def compute_empirical_threshold(
    confidences: np.ndarray,
    embeddings: np.ndarray,
    n_shuffles: int = 200,
    percentile: float = 95,
    rng: np.random.Generator | None = None,
) -> tuple[float, list[float]]:
    """Compute empirical spike detection threshold via certainty-embedding shuffle.

    Uses certainty weights w_i = 2|c_i - 0.5| (matching build_knowledge_matrix).
    Shuffles certainty-embedding pairings to destroy any real alignment,
    then records the max eigenvalue of the resulting covariance. The threshold
    is the ``percentile``-th percentile of that null distribution.

    Pass an explicit ``rng`` (np.random.Generator) for deterministic, isolated
    sampling. Defaults to a fresh ``default_rng()`` (still independent of the
    numpy global RNG state).

    Returns (threshold, null_max_eigenvalues).
    """
    if rng is None:
        rng = np.random.default_rng()
    H_t = len(confidences)
    certainty = 2 * np.abs(confidences - 0.5)
    max_eigenvalues: list[float] = []

    for _ in range(n_shuffles):
        perm = rng.permutation(H_t)
        K_shuffled = certainty[perm, np.newaxis] * embeddings
        K_centered = K_shuffled - K_shuffled.mean(axis=0, keepdims=True)
        Sigma = K_centered.T @ K_centered / H_t
        max_eigenvalues.append(float(np.linalg.eigvalsh(Sigma)[-1]))

    threshold = float(np.percentile(max_eigenvalues, percentile))
    return threshold, max_eigenvalues


def spike_p_value(eigenvalue: float, null_max_eigenvalues: list[float]) -> float:
    """Fraction of null samples with max eigenvalue >= observed."""
    return float(np.mean([m >= eigenvalue for m in null_max_eigenvalues]))


# ---------------------------------------------------------------------------
# Full spectral pipeline (Algorithm 1, lines 23-30)
# ---------------------------------------------------------------------------


def run_spectral_pipeline(
    knowledge_bank: list[HypothesisRecord],
    prev_state: Optional[SpectralState] = None,
    tw_quantile: float = 0.979,
    threshold_mode: str = "empirical",
    rng: np.random.Generator | None = None,
) -> Optional[SpectralState]:
    """Run the full end-of-generation spectral pipeline.

    Steps (Algorithm 1):
    23. Build K_t
    24. Center: K_tilde = K_t - (1/H_t)*11^T*K_t
    25. SVD of K_tilde
    26. Sigma_t = (1/H_t)*K_tilde^T*K_tilde; eigenvalues = sigma^2 / H_t
    27. MP parameters: gamma_t, sigma^2, lambda_pm
    28. TW correction -> lambda_plus_corrected
    29. Detect spikes -> J_t
    30. Compute signals S1-S4, store V_k and mean_row

    Returns None if fewer than 2 active hypotheses.
    """
    if len(knowledge_bank) < 2:
        return None

    # Step 23: build K_t
    K = build_knowledge_matrix(knowledge_bank)
    H_t, d = K.shape

    # Step 24: center
    K_tilde, mean_row = center_matrix(K)

    # Step 25-26: SVD -> eigenvalues of Sigma_t
    eigenvalues, sigma_vals, V = compute_covariance_eigenvalues(K_tilde, H_t)

    # Step 27: MP parameters
    sigma_sq = estimate_noise_variance(eigenvalues, d / H_t)
    gamma_t, lambda_plus, lambda_minus = compute_mp_parameters(d, H_t, sigma_sq)

    # Step 28: TW correction (kept as diagnostic)
    lambda_plus_corrected = compute_tw_correction(lambda_plus, H_t, tw_quantile)

    # Step 29: empirical null threshold for spike detection
    confidences = np.array([h.confidence for h in knowledge_bank])
    embeddings = np.array([h.embedding for h in knowledge_bank])
    empirical_thresh, null_max_eigs = compute_empirical_threshold(confidences, embeddings, rng=rng)

    # Choose active threshold based on mode
    if threshold_mode == "mp":
        active_threshold = lambda_plus_corrected
        active_mode = "mp"
    elif threshold_mode == "hybrid":
        # Use MP only when empirical null is clearly non-discriminative:
        # empirical threshold is > 1.5x the MP threshold
        empirical_is_degenerate = empirical_thresh > 1.5 * lambda_plus_corrected
        active_threshold = lambda_plus_corrected if empirical_is_degenerate else empirical_thresh
        active_mode = "mp" if empirical_is_degenerate else "empirical"
    else:  # "empirical" (default)
        active_threshold = empirical_thresh
        active_mode = "empirical"

    # Spike detection using active threshold
    spike_idx = np.where(eigenvalues > active_threshold)[0]
    num_spikes = len(spike_idx)
    k_t = num_spikes

    log.info(
        "Analytic MP edge: %.4f, Empirical threshold: %.4f, Active threshold: %.4f (mode=%s), Spikes detected: %d",
        lambda_plus_corrected,
        empirical_thresh,
        active_threshold,
        threshold_mode,
        k_t,
    )

    # Step 30: V_k = top-k right singular vectors.
    # Phase 3.10: widen V_k to include BBP-actionable spikes even when the v1
    # empirical-null threshold is more conservative, so downstream cluster
    # selection has something to project onto. This does NOT change v1 spike
    # count or guidance gating — those still consume k_t / spike_count.
    try:
        from esn.core.spectral_calibration import analyze_spectrum as _analyze_for_V

        _bbp_report = _analyze_for_V(
            eigenvalues,
            sigma2=float(sigma_sq),
            gamma=float(gamma_t),
            n_obs=int(len(knowledge_bank)),
        )
        _bbp_width = sum(1 for s in _bbp_report.spikes if s.above_gate)
    except Exception:  # noqa: BLE001
        _bbp_width = 0
    V_k_width = max(k_t, _bbp_width)
    V_k = V[:, :V_k_width] if V_k_width > 0 and V is not None else None

    # Effective rank
    erank = compute_effective_rank(eigenvalues)

    # Spectral signals (need previous state)
    S1 = 0.0
    S2 = 0.0
    S3 = compute_S3(eigenvalues, empirical_thresh)
    S4 = 0.0

    if prev_state is not None:
        S1 = compute_S1(num_spikes, prev_state.num_spikes)
        S2 = compute_S2(erank, prev_state.erank)
        S4 = compute_S4(
            eigenvalues,
            prev_state.eigenvalues,
            gamma_t,
            prev_state.gamma_t,
            sigma_sq,
            prev_state.sigma_sq,
        )

    # Phase 1 BBP gating: compute per-spike alignment² so the guidance builder
    # can suppress directions that are above the MP edge but whose asymptotic
    # eigenvector alignment is too low to be a reliable signal.
    from esn.core.spectral_calibration import analyze_spectrum

    spike_alignments: list[float] | None = None
    bbp_undersampled = False
    try:
        report = analyze_spectrum(
            eigenvalues,
            sigma2=float(sigma_sq),
            gamma=float(gamma_t),
            n_obs=int(len(knowledge_bank)),
        )
        spike_alignments = [s.alignment_sq for s in report.spikes]
        bbp_undersampled = bool(report.undersampled)
    except Exception:  # noqa: BLE001 - guidance gating is purely additive
        spike_alignments = None
        bbp_undersampled = False

    # Compute mutation guidance
    prev_erank = prev_state.erank if prev_state is not None else None
    guidance = compute_spectral_guidance(
        K_centered=K_tilde,
        V_k=V_k,
        hypotheses=knowledge_bank,
        spike_count=num_spikes,
        erank=erank,
        prev_erank=prev_erank,
        S1=S1,
        S2=S2,
        spike_alignments=spike_alignments,
        undersampled=bbp_undersampled,
    )

    # Diagnostic ratios
    max_eig = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0

    return SpectralState(
        eigenvalues=eigenvalues,
        V_k=V_k,
        mean_row=mean_row,
        gamma_t=gamma_t,
        sigma_sq=sigma_sq,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        lambda_plus_corrected=lambda_plus_corrected,
        num_spikes=num_spikes,
        analytic_mp_edge=lambda_plus_corrected,
        empirical_threshold=empirical_thresh,
        null_max_eigenvalues=null_max_eigs,
        mp_threshold=lambda_plus_corrected,
        empirical_mp_ratio=empirical_thresh / lambda_plus_corrected
        if lambda_plus_corrected > 1e-12
        else 0.0,
        max_eigenvalue=max_eig,
        max_eigen_empirical_ratio=max_eig / empirical_thresh if empirical_thresh > 1e-12 else 0.0,
        max_eigen_mp_ratio=max_eig / lambda_plus_corrected
        if lambda_plus_corrected > 1e-12
        else 0.0,
        S1=S1,
        S2=S2,
        S3=S3,
        S4=S4,
        erank=erank,
        mutation_guidance=guidance,
        active_threshold_mode=active_mode,
    )


# ---------------------------------------------------------------------------
# Spectral guidance for mutation prompt (Section 5.2)
# ---------------------------------------------------------------------------


def compute_spectral_guidance(
    K_centered: np.ndarray,  # (H_t, d) centered knowledge matrix
    V_k: Optional[np.ndarray],  # (d, k) top-k right singular vectors
    hypotheses: list[HypothesisRecord],  # all active hypotheses, same order as K_centered rows
    spike_count: int,
    erank: float,
    prev_erank: Optional[float],
    S1: float,  # spike emergence (change in spike count)
    S2: float,  # erank change
    spike_alignments: Optional[list[float]] = None,  # BBP alignment² per spike
    alignment_gate: float = 0.5,  # prompt-emission gate
    undersampled: bool = False,  # BBP undersampled guardrail
) -> str:
    """Translate spectral state into natural-language mutation guidance.

    Returns a string to inject into the mutation agent's prompt.
    """
    # Phase 1 follow-up: effective spike count is the union of the legacy
    # empirical detector and the BBP alignment-gated count. This keeps
    # guidance live when BBP sees actionable structure even though the
    # empirical threshold has not fired yet. BBP is suppressed in the
    # undersampled regime (n_obs<30 or gamma>0.9) where its asymptotic
    # alignment estimates are unreliable — in that case we only trust the
    # legacy empirical detector.
    bbp_actionable = 0
    if spike_alignments is not None and not undersampled:
        bbp_actionable = sum(1 for a in spike_alignments if a > alignment_gate)
    effective_spike_count = max(spike_count, bbp_actionable)

    if effective_spike_count == 0 or V_k is None:
        return "No spectral structure detected yet. Explore freely."

    guidance_parts = []

    # --- Part 1: Well-explored directions (dominant spike directions) ---
    # Phase 1 BBP gating: when per-spike alignment² is supplied, only emit
    # clusters whose BBP-asymptotic eigenvector alignment is above the gate.
    # Alignments are monotonic in lambda for spikes above the MP edge, so the
    # actionable subset is always a leading prefix.
    well_explored = []
    gated_out: list[tuple[int, float]] = []
    loop_count = min(max(effective_spike_count, spike_count), 3, V_k.shape[1])
    # In the undersampled regime BBP alignments are unreliable, so the gate
    # must not silence clusters surfaced by the legacy empirical detector.
    bbp_gate_active = spike_alignments is not None and not undersampled
    for j in range(loop_count):  # top 3 spikes
        if bbp_gate_active and j < len(spike_alignments):
            if spike_alignments[j] <= alignment_gate:
                gated_out.append((j + 1, spike_alignments[j]))
                continue
        v_j = V_k[:, j]
        projections = K_centered @ v_j  # (H_t,)
        top_indices = np.argsort(np.abs(projections))[-3:]  # top-3 hypotheses
        cluster_descriptions = []
        for idx in top_indices:
            h = hypotheses[idx]
            cluster_descriptions.append(f"  - (c={h.confidence:.2f}) {h.text[:120]}")
        align_tag = ""
        if spike_alignments is not None and j < len(spike_alignments):
            align_tag = f" (alignment²={spike_alignments[j]:.2f})"
        well_explored.append(
            f"Knowledge cluster {j + 1}{align_tag}:\n" + "\n".join(cluster_descriptions)
        )

    if well_explored:
        guidance_parts.append(
            "WELL-EXPLORED DIRECTIONS (the agent has strong theories here — "
            "mutations in these directions refine existing knowledge but may not discover new strategies):\n"
            + "\n".join(well_explored)
        )

    if gated_out:
        gated_str = ", ".join(f"cluster {idx} (alignment²={a:.2f})" for idx, a in gated_out)
        guidance_parts.append(
            f"BBP NOTE: {gated_str} suppressed — eigenvector alignment below "
            f"{alignment_gate:.2f} gate (likely finite-sample noise, not a real direction)."
        )

    # --- Part 2: Underexplored directions (high-residual hypotheses) ---
    if V_k.shape[1] > 0:
        # Project each hypothesis onto the known subspace
        projections_all = K_centered @ V_k  # (H_t, k)
        projected_back = projections_all @ V_k.T  # (H_t, d)
        residuals = K_centered - projected_back  # (H_t, d)
        residual_norms = np.linalg.norm(residuals, axis=1)  # (H_t,)
        row_norms = np.linalg.norm(K_centered, axis=1)  # (H_t,)

        # Avoid division by zero
        valid = row_norms > 1e-10
        residual_fractions = np.zeros(len(hypotheses))
        residual_fractions[valid] = residual_norms[valid] / row_norms[valid]

        # Pick hypotheses with differentiated confidence (not untested ~0.5)
        interesting_mask = np.array([h.confidence > 0.6 or h.confidence < 0.3 for h in hypotheses])

        # Combine: high residual fraction among interesting hypotheses
        scores = residual_fractions.copy()
        scores[~interesting_mask] *= 0.3  # downweight untested hypotheses

        underexplored_indices = np.argsort(scores)[-5:]  # top-5
        underexplored_descriptions = []
        for idx in underexplored_indices:
            if scores[idx] > 0.3:  # only include if meaningfully outside subspace
                h = hypotheses[idx]
                underexplored_descriptions.append(
                    f"  - (c={h.confidence:.2f}, residual={residual_fractions[idx]:.2f}) {h.text[:120]}"
                )

        if underexplored_descriptions:
            guidance_parts.append(
                "UNDEREXPLORED DIRECTIONS (these hypotheses point outside the dominant knowledge clusters — "
                "mutations exploring these ideas may discover new strategies):\n"
                + "\n".join(underexplored_descriptions)
            )

    # --- Part 3: Diversity diagnostic ---
    if S1 > 0:
        guidance_parts.append(
            f"NEW CLUSTER EMERGED: {S1:.0f} new knowledge cluster(s) just crossed the detection threshold. "
            "Consider generating mutations that test and extend this emerging theory."
        )

    if prev_erank is not None and S2 < -0.5:
        guidance_parts.append(
            "WARNING — DIVERSITY DECLINING: The knowledge structure is becoming more concentrated. "
            "The agent's understanding is narrowing. Try fundamentally different approaches "
            "that don't fit the dominant strategies."
        )

    if prev_erank is not None and S2 > 0.5:
        guidance_parts.append(
            "DIVERSITY INCREASING: New independent directions of understanding are forming. "
            "Continue the current exploration strategy."
        )

    if not guidance_parts:
        return "Spectral analysis active but no strong guidance signals this generation."

    return "\n\n".join(guidance_parts)
