# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Embedding models for ESN hypothesis vectorization."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Abstract embedding model protocol."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by this model."""
        ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of text strings. Returns shape (len(texts), dimension)."""
        ...


class SentenceTransformerEmbedder(EmbeddingModel):
    """Real embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)
