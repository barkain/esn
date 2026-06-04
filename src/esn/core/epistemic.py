# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Epistemic novelty: Beta-Bernoulli hypothesis updates and scoring.

Implements Section 3 of the ESN paper:
- Eq 5: Beta-Bernoulli confidence update
- Eq 6: Per-hypothesis delta
- Eq 7: Epistemic novelty score N_ep(x)
"""


def update_hypothesis(
    confidence: float,
    n_obs: int,
    evidence: int,
) -> tuple[float, int, float]:
    """Beta-Bernoulli confidence update (Eq 5) and delta (Eq 6).

    Args:
        confidence: Current confidence c_i in [0, 1].
        n_obs: Current observation count n_i.
        evidence: Binary evidence e_i in {0, 1}.

    Returns:
        (new_confidence, new_n_obs, delta)
    """
    # Eq 5: c'_i = (c_i * n_i + e_i) / (n_i + 1)
    c_old = confidence
    c_new = (c_old * n_obs + evidence) / (n_obs + 1)
    n_new = n_obs + 1

    # Eq 6: delta_i = |e_i - c_i| / (n_i + 1)
    delta = abs(evidence - c_old) / (n_obs + 1)

    return c_new, n_new, delta


def compute_epistemic_novelty(
    relevant_hypotheses: list[dict],
    new_hypothesis_count: int = 0,
    prediction_surprise: bool = False,
    alpha: float = 0.1,
    beta: float = 0.05,
    actual_score: float | None = None,
    failure_threshold: float = 0.0,
    failure_discount: float = 0.3,
) -> float:
    """Compute epistemic novelty score N_ep(x) per Eq 7 with failure discount.

    N_ep(x) = SUM_{i in R} c_i^old * delta_i
              + alpha * |H_new|
              + beta * 1[y not in [y_lo, y_hi]]

    When actual_score <= failure_threshold, the raw score is multiplied by
    failure_discount so that broken solutions are deprioritized for parent
    selection while still contributing hypotheses.

    Each entry in relevant_hypotheses must contain:
        - "confidence": c_i (the OLD confidence, before update)
        - "delta": delta_i computed via update_hypothesis

    Args:
        relevant_hypotheses: List of dicts with "confidence" and "delta".
        new_hypothesis_count: |H_new| from post-evaluation analysis.
        prediction_surprise: Whether actual score fell outside predicted range.
        alpha: Weight for new hypotheses (paper: 0.1).
        beta: Weight for prediction surprise (paper: 0.05).
        actual_score: The solution's objective score (None to skip discount).
        failure_threshold: Scores at or below this are considered failures.
        failure_discount: Multiplicative discount applied to failed solutions.

    Returns:
        Epistemic novelty score (non-negative, unbounded above).
    """
    # Term (a): hypothesis revision
    revision_score = sum(h["confidence"] * h["delta"] for h in relevant_hypotheses)

    # Term (b): new hypotheses
    new_hyp_bonus = alpha * new_hypothesis_count

    # Term (c): prediction surprise
    surprise_bonus = beta * (1.0 if prediction_surprise else 0.0)

    raw_score = revision_score + new_hyp_bonus + surprise_bonus

    # Failure-aware discount (Eq 7 modification)
    if actual_score is not None and actual_score <= failure_threshold:
        raw_score *= failure_discount

    return raw_score


def normalize_epistemic(raw_novelty: float, max_observed: float = 1.0) -> float:
    """Normalize N_ep to [0, 1] for use in the unified score (Eq 13).

    Uses min-max normalization against the running maximum.
    If max_observed is 0, returns 0.
    """
    if max_observed <= 0.0:
        return 0.0
    return min(1.0, raw_novelty / max_observed)
