# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Public ``esn.run()`` novelty wiring: analyzer/predictor params + loud warning.

These lock the fix for the gap where ``esn.run()`` left ESN's epistemic-spectral
novelty machinery unreachable: novelty now activates by supplying an ``analyzer``
(there is no ``use_novelty`` flag), and a run without one warns loudly instead of
silently degrading to fitness-only search.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

import esn
import esn.api as api


class _SpyEngine:
    """Records constructor kwargs; no-ops the generation loop."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _SpyEngine.last_kwargs = kwargs
        self._best_score = 1.0
        self._best_code = "def solve():\n    return 0.0\n"

    def run_batch_generation(self):
        return []

    def run_generation(self):
        return None


@pytest.fixture
def spy_engine(monkeypatch):
    import esn.engine.engine as eng

    monkeypatch.setattr(eng, "ESNEngine", _SpyEngine)
    _SpyEngine.last_kwargs = {}
    return _SpyEngine


def _domain():
    return SimpleNamespace(name="t")


def test_run_without_analyzer_warns_loudly(spy_engine):
    with pytest.warns(RuntimeWarning, match="novelty machinery is INACTIVE"):
        api.run(_domain(), generations=1, batch_size=2)
    # Fitness-only: no novelty stack built, analyzer forwarded as None.
    assert spy_engine.last_kwargs["analyzer"] is None
    assert spy_engine.last_kwargs["novelty_computer"] is None
    assert spy_engine.last_kwargs["knowledge"] is None


def test_run_with_analyzer_activates_and_forwards(monkeypatch, spy_engine):
    # Stub the stack builder so the test doesn't load the heavy embedder.
    monkeypatch.setattr(
        api, "_build_novelty_stack", lambda seed, mode="empirical": ("KB", "NC", "CFG")
    )
    analyzer = object()
    predictor = object()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the INACTIVE warning must NOT fire
        api.run(
            _domain(),
            generations=1,
            batch_size=2,
            analyzer=analyzer,
            predictor=predictor,
        )
    k = spy_engine.last_kwargs
    assert k["analyzer"] is analyzer
    assert k["predictor"] is predictor
    assert (k["knowledge"], k["novelty_computer"], k["config"]) == ("KB", "NC", "CFG")


def test_run_forwards_recombination_and_threshold_mode(monkeypatch, spy_engine):
    captured = {}

    def fake_stack(seed, spectral_threshold_mode="empirical"):
        captured["mode"] = spectral_threshold_mode
        return ("KB", "NC", "CFG")

    monkeypatch.setattr(api, "_build_novelty_stack", fake_stack)
    api.run(
        _domain(),
        generations=1,
        batch_size=2,
        analyzer=object(),
        enable_recombination=True,
        spectral_threshold_mode="hybrid",
    )
    assert spy_engine.last_kwargs["enable_recombination"] is True
    assert captured["mode"] == "hybrid"


def test_use_novelty_kwarg_is_removed(spy_engine):
    with pytest.raises(TypeError):
        api.run(_domain(), generations=1, use_novelty=True)


def test_factories_reexported_and_validate_provider():
    assert callable(esn.make_analyzer)
    assert callable(esn.make_predictor)
    with pytest.raises(ValueError):
        esn.make_analyzer(model="bogus-model-xyz")
    with pytest.raises(ValueError):
        esn.make_predictor(model="bogus-model-xyz")
