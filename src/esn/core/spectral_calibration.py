# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""BBP-grounded per-spike calibration of a spectral state.

This module is pure math. It has no dependencies on the rest of the engine
beyond ``numpy`` so that it can be unit-tested against persisted spectral
inputs without exercising the full pipeline.

Design rationale: encoder-agnostic spectral calibration via BBP-grounded
spike interpretation. In short:

* Detect candidate spikes via the analytic Marchenko-Pastur edge — this is
  the right necessary condition.
* Qualify each candidate by inverting the BBP fixed-point relation
  ``lambda_hat = (1 + theta)(1 + gamma/theta)`` to recover the population
  spike strength ``theta``, then compute the asymptotic eigenvector
  alignment squared ``|<v_hat, v>|^2 = (1 - gamma/theta^2) / (1 + gamma/theta)``.
* Use ``alignment_i^2`` as the per-direction reliability score the
  controller should consume — not raw spike counts.
* Layer a finite-sample ``undersampled`` guardrail on top because BBP is
  asymptotic and ESN runs are small-``n``.

Reference: Baik, Ben Arous, Péché (2005); Barbier RMT/BBP lecture notes
(https://jeanbarbier.github.io/jeanbarbier/docs/part1_rmt_bbp_beamer.pdf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Default per-spike alignment cutoff used by the controller-emission gate.
# A spike with alignment^2 below this is counted as a BBP candidate but is
# not surfaced to the mutator as an actionable direction.
DEFAULT_ALIGNMENT_GATE: float = 0.5


# ---------------------------------------------------------------------------
# Core BBP math
# ---------------------------------------------------------------------------


def mp_edge(sigma2: float, gamma: float) -> float:
    """Upper edge of the Marchenko-Pastur bulk for noise variance ``sigma2``
    and aspect ratio ``gamma = p / n``.
    """
    if sigma2 < 0 or gamma < 0:
        raise ValueError("sigma2 and gamma must be non-negative")
    return float(sigma2 * (1.0 + math.sqrt(gamma)) ** 2)


def bbp_critical_theta(gamma: float) -> float:
    """The BBP critical population spike strength: a spike with theta above
    ``sqrt(gamma)`` (in unit-variance convention) escapes the bulk.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return math.sqrt(gamma)


def bbp_invert(lambda_hat: float, sigma2: float, gamma: float) -> float | None:
    """Recover the population spike strength ``theta`` from an observed
    sample eigenvalue ``lambda_hat``, by inverting the BBP relation.

    Working in the unit-variance convention with ``x = lambda_hat / sigma2``,
    BBP says ``x = (1 + theta) (1 + gamma / theta)``, which rearranges to::

        theta^2 + theta * (1 + gamma - x) + gamma = 0

    The two real roots straddle ``sqrt(gamma)``; we return the larger root
    (the supercritical one) when the spike is above the BBP threshold, and
    ``None`` when the eigenvalue lies inside the bulk.

    Returns ``None`` when:
        * ``lambda_hat`` is at or below the analytic MP upper edge, or
        * the resulting quadratic has no real roots, or
        * the supercritical root is at or below ``sqrt(gamma)``.
    """
    if sigma2 <= 0:
        return None
    if gamma <= 0:
        # Degenerate fully-sampled regime: every nonzero eigenvalue is a spike,
        # and BBP collapses. Treat ``theta = (lambda_hat / sigma2) - 1`` as
        # the trivial inversion when ``lambda_hat > sigma2``.
        return max(lambda_hat / sigma2 - 1.0, 0.0) or None

    # Necessary condition: must be above MP edge to be a candidate spike.
    if lambda_hat <= mp_edge(sigma2, gamma):
        return None

    x = lambda_hat / sigma2
    b = 1.0 + gamma - x
    disc = b * b - 4.0 * gamma
    if disc < 0:
        return None
    theta = (-b + math.sqrt(disc)) / 2.0
    crit = bbp_critical_theta(gamma)
    if theta <= crit:
        return None
    return float(theta)


def alignment_squared(theta: float, gamma: float) -> float:
    """Asymptotic squared eigenvector alignment for a population spike of
    strength ``theta`` at aspect ratio ``gamma``.

    Returns 0 below the BBP threshold (``theta <= sqrt(gamma)``) and a value
    in [0, 1] above it. At ``theta = sqrt(gamma)`` the alignment is exactly 0;
    as ``theta -> infinity`` it approaches 1.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    crit = bbp_critical_theta(gamma)
    if theta <= crit:
        return 0.0
    num = 1.0 - gamma / (theta * theta)
    den = 1.0 + gamma / theta
    if den <= 0:
        return 0.0
    val = num / den
    # Numerical safety: clamp to [0, 1] for tiny FP drift near the edge.
    return float(max(0.0, min(1.0, val)))


# ---------------------------------------------------------------------------
# Finite-sample guardrails
# ---------------------------------------------------------------------------

# Defaults are conservative: BBP is asymptotic in (n, p), and these regimes
# are where ESN runs typically live.
DEFAULT_UNDERSAMPLED_NOBS: int = 30
DEFAULT_UNDERSAMPLED_GAMMA: float = 0.9


def is_undersampled(
    n_obs: int,
    gamma: float,
    min_n_obs: int = DEFAULT_UNDERSAMPLED_NOBS,
    max_gamma: float = DEFAULT_UNDERSAMPLED_GAMMA,
) -> bool:
    """Return True if the spectral regime is too small / too aspect-square
    for BBP asymptotics to be trusted at face value.

    The controller should still consume ``alignment_i^2`` as the primary
    signal, but mark its confidence as degraded and lean more on epistemic
    novelty when this flag is set.
    """
    return n_obs < min_n_obs or gamma >= max_gamma


# ---------------------------------------------------------------------------
# Spectral report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpikeInfo:
    """Per-spike BBP-derived statistics."""

    rank: int  # 0-indexed position in the sorted spectrum
    lambda_hat: float  # observed sample eigenvalue
    theta: float  # BBP-inverted population strength
    alignment_sq: float  # asymptotic squared eigenvector alignment
    above_gate: bool  # alignment_sq > controller emission gate


@dataclass(frozen=True)
class SpectralReport:
    """BBP-grounded interpretation of an observed spectrum.

    Additive over the existing ``SpectralState`` fields — this report does
    NOT replace the engine's existing num_spikes / threshold logic; it sits
    alongside as a more reliable signal source for the controller. The
    controller policy is: emit only ``SpikeInfo`` entries with ``above_gate``
    set true into the mutator prompt.
    """

    sigma2: float
    gamma: float
    n_obs: int
    mp_upper_edge: float

    spike_count_bbp: int  # # eigenvalues above MP edge
    spikes: tuple[SpikeInfo, ...]  # one per BBP candidate, ranked desc
    effective_spike_count: float  # sum of alignment^2 across spikes
    dominant_alignment: float  # alignment^2 of the leading spike (0 if none)
    leading_gap: float  # lambda_0 / max(lambda_1, eps)
    undersampled: bool  # finite-sample caution flag
    alignment_gate: float  # threshold used for above_gate

    # Extras useful for reports / debugging
    bbp_critical_theta: float = field(default=0.0)


def analyze_spectrum(
    eigenvalues: Sequence[float] | np.ndarray,
    sigma2: float,
    gamma: float,
    n_obs: int,
    *,
    alignment_gate: float = DEFAULT_ALIGNMENT_GATE,
    undersampled_min_n_obs: int = DEFAULT_UNDERSAMPLED_NOBS,
    undersampled_max_gamma: float = DEFAULT_UNDERSAMPLED_GAMMA,
) -> SpectralReport:
    """Build a BBP-grounded report from an observed eigenvalue spectrum.

    ``eigenvalues`` may be in any order; this function sorts them descending.
    Eigenvalues at or below the analytic MP upper edge are treated as bulk
    and excluded from spike accounting (they are necessary-condition rejects).
    Eigenvalues above the MP edge are inverted via BBP to obtain ``theta``
    and ``alignment_sq``; spikes whose alignment_sq exceeds ``alignment_gate``
    are marked ``above_gate=True`` and are the directions the controller
    should surface to the mutator.
    """
    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if n_obs < 0:
        raise ValueError("n_obs must be non-negative")

    eigs = np.asarray(eigenvalues, dtype=np.float64)
    if eigs.ndim != 1:
        eigs = eigs.ravel()
    eigs = np.sort(eigs)[::-1]  # descending

    edge = mp_edge(sigma2, gamma)
    crit = bbp_critical_theta(gamma)

    spikes: list[SpikeInfo] = []
    for i, lam in enumerate(eigs):
        if lam <= edge:
            break  # rest of the spectrum is bulk or noise floor
        theta = bbp_invert(float(lam), sigma2, gamma)
        if theta is None:
            # Above MP edge but failed BBP inversion — treat as marginal
            # noise. Should be rare.
            continue
        a2 = alignment_squared(theta, gamma)
        spikes.append(
            SpikeInfo(
                rank=i,
                lambda_hat=float(lam),
                theta=float(theta),
                alignment_sq=float(a2),
                above_gate=bool(a2 > alignment_gate),
            )
        )

    leading_gap = 0.0
    if eigs.size >= 2:
        denom = float(eigs[1]) if eigs[1] > 0 else 1e-12
        leading_gap = float(eigs[0] / denom)
    elif eigs.size == 1 and eigs[0] > 0:
        leading_gap = float("inf")

    effective_spike_count = float(sum(s.alignment_sq for s in spikes))
    dominant_alignment = float(spikes[0].alignment_sq) if spikes else 0.0

    return SpectralReport(
        sigma2=float(sigma2),
        gamma=float(gamma),
        n_obs=int(n_obs),
        mp_upper_edge=float(edge),
        spike_count_bbp=len(spikes),
        spikes=tuple(spikes),
        effective_spike_count=effective_spike_count,
        dominant_alignment=dominant_alignment,
        leading_gap=leading_gap,
        undersampled=is_undersampled(
            n_obs,
            gamma,
            min_n_obs=undersampled_min_n_obs,
            max_gamma=undersampled_max_gamma,
        ),
        alignment_gate=float(alignment_gate),
        bbp_critical_theta=float(crit),
    )


# ---------------------------------------------------------------------------
# Convenience: actionable spike subset for the controller
# ---------------------------------------------------------------------------


def actionable_spikes(report: SpectralReport) -> tuple[SpikeInfo, ...]:
    """The subset of spikes the controller should surface to the mutator."""
    return tuple(s for s in report.spikes if s.above_gate)


def report_to_dict(report: SpectralReport) -> dict:
    """Plain-dict view for serialization into report.md / engine state."""
    return {
        "sigma2": report.sigma2,
        "gamma": report.gamma,
        "n_obs": report.n_obs,
        "mp_upper_edge": report.mp_upper_edge,
        "bbp_critical_theta": report.bbp_critical_theta,
        "spike_count_bbp": report.spike_count_bbp,
        "effective_spike_count": report.effective_spike_count,
        "dominant_alignment": report.dominant_alignment,
        "leading_gap": report.leading_gap,
        "undersampled": report.undersampled,
        "alignment_gate": report.alignment_gate,
        "spikes": [
            {
                "rank": s.rank,
                "lambda_hat": s.lambda_hat,
                "theta": s.theta,
                "alignment_sq": s.alignment_sq,
                "above_gate": s.above_gate,
            }
            for s in report.spikes
        ],
    }
