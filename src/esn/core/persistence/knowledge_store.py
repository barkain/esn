# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Persistence for KnowledgeIntegration hypothesis bank."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from esn.core.spectral_models import ESNConfig, HypothesisRecord
from esn.core.knowledge import KnowledgeIntegration


def _ndarray_to_b64(arr: np.ndarray) -> dict:
    """Serialize numpy array to base64 with shape/dtype metadata."""
    return {
        "data": base64.b64encode(arr.tobytes()).decode(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _b64_to_ndarray(d: dict) -> np.ndarray:
    """Deserialize numpy array from base64 with shape/dtype metadata."""
    arr = np.frombuffer(base64.b64decode(d["data"]), dtype=np.dtype(d["dtype"]))
    return arr.reshape(d["shape"]).copy()


class KnowledgeStore:
    """JSON-based save/load for KnowledgeIntegration state."""

    @staticmethod
    def save(knowledge: KnowledgeIntegration, path: Path) -> None:
        """Persist all hypotheses from the knowledge bank."""
        hypotheses = []
        for h in knowledge.bank.get_all_hypotheses():
            hypotheses.append(
                {
                    "id": h.id,
                    "text": h.text,
                    "confidence": h.confidence,
                    "n_obs": h.n_obs,
                    "embedding": _ndarray_to_b64(h.embedding),
                    "concepts": h.concepts,
                    "created_at": h.created_at,
                    "last_tested": h.last_tested,
                    "status": h.status,
                }
            )
        data = {"hypotheses": hypotheses}
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(
        path: Path,
        config: ESNConfig | None = None,
        embedder: Any | None = None,
        embedding_dim: int = 1024,
    ) -> KnowledgeIntegration:
        """Restore KnowledgeIntegration from a saved file."""
        ki = KnowledgeIntegration(config=config, embedder=embedder, embedding_dim=embedding_dim)
        if not path.exists():
            return ki
        data = json.loads(path.read_text())
        for item in data.get("hypotheses", []):
            record = HypothesisRecord(
                id=item["id"],
                text=item["text"],
                confidence=item["confidence"],
                n_obs=item["n_obs"],
                embedding=_b64_to_ndarray(item["embedding"]),
                concepts=item["concepts"],
                created_at=item["created_at"],
                last_tested=item["last_tested"],
                status=item["status"],
            )
            ki.bank.add(record)
        return ki
