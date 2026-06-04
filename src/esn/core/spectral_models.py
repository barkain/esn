# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Core data models for ESN framework."""

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional


class HypothesisStatus(StrEnum):
    """Status of a hypothesis in its lifecycle."""

    ACTIVE = "active"
    RETIRED = "retired"
    ARCHIVED = "archived"


class HypothesisRecord(BaseModel):
    """Represents a causal hypothesis with confidence and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    text: str
    confidence: float = 0.5
    n_obs: int = 1
    embedding: np.ndarray
    concepts: list[str]
    created_at: int
    last_tested: int
    status: Literal["active", "retired", "archived"] = "active"


class LedgerEntry(BaseModel):
    """Records a solution attempt and its epistemic impact."""

    model_config = ConfigDict(extra="ignore")

    generation: int
    solution_id: str
    solution_code: str
    parent_id: Optional[str] = None
    strategy: str
    predicted_range: tuple[float, float]
    actual_score: float
    evidence: list[dict]
    new_hypotheses: list[dict]
    epistemic_novelty: float
    spectral_novelty: Optional[float] = None
    unified_novelty: Optional[float] = None


class SpectralState(BaseModel):
    """Stores the result of the end-of-generation spectral pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core spectral data
    eigenvalues: np.ndarray  # eigenvalues of Sigma_t (descending)
    V_k: Optional[np.ndarray] = None  # top-k right singular vectors of K_tilde
    mean_row: Optional[np.ndarray] = None  # mean row of K_t (for centering)

    # MP parameters
    gamma_t: float = 0.0  # aspect ratio d / H_t
    sigma_sq: float = 0.0  # estimated noise variance
    lambda_plus: float = 0.0  # MP upper edge
    lambda_minus: float = 0.0  # MP lower edge
    lambda_plus_corrected: float = 0.0  # TW-corrected upper edge

    # Spike detection
    num_spikes: int = 0

    # Empirical null threshold
    analytic_mp_edge: float = 0.0  # TW-corrected MP edge (diagnostic only)
    empirical_threshold: Optional[float] = None  # shuffle-null 95th percentile
    null_max_eigenvalues: Optional[list[float]] = None  # for p-value computation

    # Threshold diagnostics
    mp_threshold: float = 0.0  # TW-corrected MP edge
    empirical_mp_ratio: float = (
        0.0  # empirical_threshold / mp_threshold (>1 means empirical is more conservative)
    )
    max_eigenvalue: float = 0.0  # largest eigenvalue
    max_eigen_empirical_ratio: float = 0.0  # max_eigenvalue / empirical_threshold
    max_eigen_mp_ratio: float = 0.0  # max_eigenvalue / mp_threshold

    # Spectral signals
    S1: float = 0.0  # spike emergence
    S2: float = 0.0  # effective rank change
    S3: Optional[float] = None  # power-law exponent (None if < 3 spikes)
    S4: float = 0.0  # spectral divergence from null

    # Derived
    erank: float = 0.0  # effective rank of Sigma_t

    # Mutation guidance (natural language for next generation's mutation prompt)
    mutation_guidance: str = ""

    # Which threshold was actually used: "empirical" or "mp"
    active_threshold_mode: str = "empirical"

    # Compression metadata
    spectral_dim: int = 384  # Working dimension used for this state
    observation_count: int = 0  # Total observations (hypotheses + expanded)


class DiagnosticVector(BaseModel):
    """Per-generation diagnostic vector D(t) = (S1, S2, alpha_t, S4, gamma_t)."""

    S1: float = 0.0
    S2: float = 0.0
    alpha_t: Optional[float] = None
    S4: float = 0.0
    gamma_t: float = 0.0


class ESNConfig(BaseModel):
    """Configuration parameters for ESN framework.

    Paper reference values:
    - alpha (Eq 7): weight for new hypothesis count, default 0.1
    - beta (Eq 7): weight for prediction surprise, default 0.05
    - tau (Eq 14): temperature for adaptive mixing, default 5.0
    - tw_quantile: F_TW1^{-1}(0.95) approx 0.979
    - spectral_p: p-value for Tracy-Widom correction, default 0.05
    """

    embedding_dim: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Epistemic novelty parameters (Eq 7)
    alpha: float = 0.1  # new hypothesis bonus weight
    beta: float = 0.05  # prediction surprise weight
    # Spectral / unified parameters
    tau: float = 5.0  # adaptive mixing temperature (Eq 14)
    tw_quantile: float = 0.979  # F_TW1^{-1}(0.95)
    spectral_p: float = 0.05  # Tracy-Widom p-value
    # Failure-aware epistemic novelty
    failure_discount: float = 0.3  # discount factor for failed solutions
    failure_threshold: float = 0.0  # scores <= this are considered failures
    # Retirement
    retirement_threshold: float = 0.1
    retirement_min_obs: int = 10
    # Code size budget
    max_code_size: int = 5000  # max recommended solution chars
    # Population
    population_size: int = 1  # candidates per generation
    # Selection strategy
    selection_strategy: str = (
        "pareto"  # pareto, pareto_fitness, pareto_unified, weighted_sum, novelty_gated
    )
    fitness_epsilon: float = 0.05  # epsilon-band for pareto selection
    novelty_weight: float = 0.3  # for weighted_sum strategy
    novelty_threshold: float = 0.5  # for novelty_gated strategy
    # Signal-quality gate for adaptive mixing (§2)
    min_spike_persistence: int = 3  # consecutive gens with spikes before spectral weight activates
    # Stagnation / search temperature
    stagnation_threshold: int = 3  # gens without improvement before temperature rises
    max_temperature: float = 1.0  # maximum search temperature
    temperature_increment: float = 0.15  # temperature increase per stagnant generation
    # Per-task model routing
    mutation_model: str = "claude-opus-4-6"  # Strong reasoning for code gen (via SDK)
    prediction_model: str = "gpt-4o-mini"  # Fast JSON prediction (via OpenAI)
    analysis_model: str = "gpt-4o-mini"  # Evidence analysis (via OpenAI)
    maintenance_model: str = "gpt-4o-mini"  # Mechanical operations (via OpenAI)
    # Meta-question hypothesis generation
    meta_questions_per_gen: int = 2  # 0 to disable; max analytical questions per evaluation
    max_new_hypotheses_per_eval: int = 5  # cap new hypotheses per candidate evaluation
    # Hypothesis admission control
    admission_cosine_threshold: float = (
        0.88  # embedding cosine above which a new hyp is a duplicate
    )
    admission_tag_overlap: float = 0.3  # normalized tag Jaccard threshold for duplicate check
    # Hypothesis TTL — auto-retire untested hypotheses after N generations
    hypothesis_ttl: int = 5  # retire if n_obs==1 after this many gens since creation (0=disabled)
    # Spectral compression
    spectral_dim: int = 48  # Working dimension for spectral pipeline (PCA compression)
    # Spectral threshold mode
    spectral_threshold_mode: str = "empirical"  # "empirical" | "mp" | "hybrid"
    # Time budget
    mutation_timeout: int = 180  # seconds per mutation call before fallback
    eval_timeout: int = 30  # seconds per solution evaluation
