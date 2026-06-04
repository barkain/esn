# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Persistence for NoveltyComputer spectral and epistemic state."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from esn.core.spectral_models import ESNConfig, SpectralState
from esn.core.knowledge import KnowledgeIntegration
from esn.core.novelty import NoveltyComputer


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


def _serialize_spectral_state(state: SpectralState) -> dict:
    """Convert SpectralState to JSON-serializable dict."""
    data: dict[str, Any] = {}
    # Numpy arrays
    data["eigenvalues"] = _ndarray_to_b64(state.eigenvalues)
    data["V_k"] = _ndarray_to_b64(state.V_k) if state.V_k is not None else None
    data["mean_row"] = _ndarray_to_b64(state.mean_row) if state.mean_row is not None else None
    # Scalar floats
    for field in (
        "gamma_t",
        "sigma_sq",
        "lambda_plus",
        "lambda_minus",
        "lambda_plus_corrected",
        "analytic_mp_edge",
        "S1",
        "S2",
        "S4",
        "erank",
    ):
        data[field] = getattr(state, field)
    # Optional floats
    data["num_spikes"] = state.num_spikes
    data["empirical_threshold"] = state.empirical_threshold
    data["S3"] = state.S3
    data["null_max_eigenvalues"] = state.null_max_eigenvalues
    # String
    data["mutation_guidance"] = state.mutation_guidance
    # Threshold diagnostics
    data["active_threshold_mode"] = state.active_threshold_mode
    data["mp_threshold"] = state.mp_threshold
    data["empirical_mp_ratio"] = state.empirical_mp_ratio
    data["max_eigenvalue"] = state.max_eigenvalue
    data["max_eigen_empirical_ratio"] = state.max_eigen_empirical_ratio
    data["max_eigen_mp_ratio"] = state.max_eigen_mp_ratio
    # Compression metadata
    data["spectral_dim"] = state.spectral_dim
    data["observation_count"] = state.observation_count
    return data


def _deserialize_spectral_state(data: dict) -> SpectralState:
    """Reconstruct SpectralState from serialized dict."""
    eigenvalues = _b64_to_ndarray(data["eigenvalues"])
    V_k = _b64_to_ndarray(data["V_k"]) if data.get("V_k") is not None else None
    mean_row = _b64_to_ndarray(data["mean_row"]) if data.get("mean_row") is not None else None
    return SpectralState(
        eigenvalues=eigenvalues,
        V_k=V_k,
        mean_row=mean_row,
        gamma_t=data.get("gamma_t", 0.0),
        sigma_sq=data.get("sigma_sq", 0.0),
        lambda_plus=data.get("lambda_plus", 0.0),
        lambda_minus=data.get("lambda_minus", 0.0),
        lambda_plus_corrected=data.get("lambda_plus_corrected", 0.0),
        num_spikes=data.get("num_spikes", 0),
        analytic_mp_edge=data.get("analytic_mp_edge", 0.0),
        empirical_threshold=data.get("empirical_threshold"),
        null_max_eigenvalues=data.get("null_max_eigenvalues"),
        S1=data.get("S1", 0.0),
        S2=data.get("S2", 0.0),
        S3=data.get("S3"),
        S4=data.get("S4", 0.0),
        erank=data.get("erank", 0.0),
        mutation_guidance=data.get("mutation_guidance", ""),
        # Threshold diagnostics
        active_threshold_mode=data.get("active_threshold_mode", "empirical"),
        mp_threshold=data.get("mp_threshold", 0.0),
        empirical_mp_ratio=data.get("empirical_mp_ratio", 0.0),
        max_eigenvalue=data.get("max_eigenvalue", 0.0),
        max_eigen_empirical_ratio=data.get("max_eigen_empirical_ratio", 0.0),
        max_eigen_mp_ratio=data.get("max_eigen_mp_ratio", 0.0),
        # Compression metadata
        spectral_dim=data.get("spectral_dim", 384),
        observation_count=data.get("observation_count", 0),
    )


class NoveltyStore:
    """JSON-based save/load for NoveltyComputer state."""

    @staticmethod
    def save(novelty: NoveltyComputer, path: Path) -> None:
        """Persist spectral state, spike history, and max epistemic."""
        data: dict[str, Any] = {
            "spike_count_history": novelty._spike_count_history,
            "max_epistemic": novelty._max_epistemic,
        }
        if novelty._spectral_state is not None:
            data["spectral_state"] = _serialize_spectral_state(novelty._spectral_state)
        else:
            data["spectral_state"] = None
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(
        path: Path,
        knowledge: KnowledgeIntegration,
        config: ESNConfig | None = None,
    ) -> NoveltyComputer:
        """Restore NoveltyComputer from a saved file."""
        nc = NoveltyComputer(knowledge=knowledge, config=config)
        if not path.exists():
            return nc
        data = json.loads(path.read_text())
        nc._spike_count_history = data.get("spike_count_history", [])
        nc._max_epistemic = data.get("max_epistemic", 0.0)
        ss = data.get("spectral_state")
        if ss is not None:
            nc._spectral_state = _deserialize_spectral_state(ss)
        return nc
