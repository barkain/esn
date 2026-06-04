# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Persistence for search state."""

from __future__ import annotations

from pathlib import Path

from esn.core.models import SearchState


class SearchStateStore:
    """JSON-based save/load for SearchState."""

    @staticmethod
    def save(state: SearchState, path: Path) -> None:
        path.write_text(state.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path) -> SearchState:
        if not path.exists():
            return SearchState()
        return SearchState.model_validate_json(path.read_text())
