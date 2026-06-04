# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Utility functions for ESN framework."""

import numpy as np


def generate_fallback_embedding(text: str, dim: int = 1024) -> np.ndarray:
    """Generate a deterministic fallback embedding from text.

    Used when no real embedder is available (e.g. during maintenance splits).
    Produces a repeatable but non-meaningful vector from the text hash.
    """
    rng = np.random.default_rng(hash(text) % (2**32))
    return rng.standard_normal(dim)


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
