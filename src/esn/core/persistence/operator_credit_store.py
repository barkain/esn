# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Persistence for operator credit model."""

from __future__ import annotations

import json
from pathlib import Path

from esn.core.operator_credit import OperatorCreditModel


class OperatorCreditStore:
    """JSON-based save/load for OperatorCreditModel."""

    @staticmethod
    def save(model: OperatorCreditModel, path: Path) -> None:
        path.write_text(json.dumps(model.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> OperatorCreditModel:
        if not path.exists():
            return OperatorCreditModel()
        data = json.loads(path.read_text())
        return OperatorCreditModel.from_dict(data)
