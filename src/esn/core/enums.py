# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Search mode enums for ESN core."""

from __future__ import annotations

from enum import Enum


class SearchMode(str, Enum):
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    REPAIR = "repair"
    COMPRESS = "compress"
    BRIDGE = "bridge"
    RECOVER = "recover"
