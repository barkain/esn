# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Spectral dimension compression and observation expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from esn.core.spectral_models import HypothesisRecord


@dataclass
class SpectralObservation:
    """An additional observation for the spectral pipeline beyond hypotheses."""

    text: str
    embedding: np.ndarray | None  # shape (embedding_dim,), or None to be embedded later
    weight: float = 0.5  # analogous to certainty, 0-1
    source: str = ""  # e.g. "family_summary", "failure_summary"


@runtime_checkable
class SpectralObservationProvider(Protocol):
    """Domain-agnostic provider of extra spectral observations."""

    def get_spectral_observations(self) -> list[SpectralObservation]: ...


class SpectralCompressor:
    """PCA-based dimension reduction for spectral pipeline."""

    def __init__(self, target_dim: int = 48):
        self.target_dim = target_dim
        self._components: np.ndarray | None = None  # shape (target_dim, d)
        self._mean: np.ndarray | None = None  # shape (d,)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit PCA on embeddings and return compressed version.

        Args:
            embeddings: shape (n, d) where d is original dim
        Returns:
            compressed: shape (n, target_dim)
        """
        if embeddings.shape[0] == 0:
            return embeddings

        n, d = embeddings.shape
        actual_dim = min(self.target_dim, n, d)

        # Center
        self._mean = embeddings.mean(axis=0)
        centered = embeddings - self._mean

        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self._components = Vt[:actual_dim]  # shape (actual_dim, d)

        # Project
        compressed = centered @ self._components.T  # shape (n, actual_dim)
        return compressed

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted PCA.

        Args:
            embeddings: shape (n, d) or (d,)
        Returns:
            compressed: shape (n, target_dim) or (target_dim,)
        """
        if self._components is None or self._mean is None:
            return embeddings  # not fitted, pass through

        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[np.newaxis, :]

        centered = embeddings - self._mean
        compressed = centered @ self._components.T

        return compressed[0] if single else compressed


def expand_observations(
    hypothesis_records: list[HypothesisRecord],
    providers: list[SpectralObservationProvider],
    embedder: Any | None = None,
    embedding_dim: int = 1024,
) -> list[HypothesisRecord]:
    """Merge hypothesis records with additional observations from providers.

    Creates synthetic HypothesisRecord objects for non-hypothesis observations
    so the existing spectral pipeline works unchanged.
    """
    all_records = list(hypothesis_records)

    for provider in providers:
        observations = provider.get_spectral_observations()
        for i, obs in enumerate(observations):
            # Embed if needed
            embedding = obs.embedding
            if embedding is None and embedder is not None:
                embedding = embedder.embed(obs.text)
            if embedding is None:
                embedding = np.zeros(embedding_dim)

            # Map weight to confidence: weight=1.0 -> confidence=1.0, weight=0.0 -> confidence=0.5
            confidence = 0.5 + 0.5 * obs.weight

            synthetic = HypothesisRecord(
                id=f"obs_{obs.source}_{i}",
                text=obs.text,
                confidence=confidence,
                n_obs=1,
                embedding=embedding,
                concepts=[],
                created_at=0,
                last_tested=0,
                status="active",
            )
            all_records.append(synthetic)

    return all_records


def compress_records(
    records: list[HypothesisRecord],
    compressor: SpectralCompressor,
) -> list[HypothesisRecord]:
    """Compress all record embeddings using the fitted compressor.

    Returns new HypothesisRecord objects with compressed embeddings.
    Original records are NOT modified.
    """
    if not records:
        return []

    embeddings = np.array([r.embedding for r in records])
    compressed_embeddings = compressor.fit_transform(embeddings)

    compressed_records = []
    for r, comp_emb in zip(records, compressed_embeddings):
        compressed = r.model_copy(update={"embedding": comp_emb})
        compressed_records.append(compressed)

    return compressed_records
