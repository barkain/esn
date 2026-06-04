# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""ESN core: critic / archive / novelty infrastructure used by the engine.

Intentionally empty of re-exports. The engine and persistence layer
import core submodules by their full path (e.g. ``from esn.core.archives
import EliteArchive``), so this package needs no eager imports — and adding
them would risk pulling optional/heavy dependencies at ``import esn.core``
time. The core planner / protocols / llm_planner surfaces are not ported to
the public package and are deliberately not referenced here.
"""
