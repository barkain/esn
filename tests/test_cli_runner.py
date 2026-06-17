# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""examples/run.py CLI: arg parsing + flag -> esn.run wiring (no real LLM run)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parents[1] / "examples" / "run.py"
_spec = importlib.util.spec_from_file_location("esn_example_run", _RUN_PY)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def test_parse_args_defaults():
    a = runner._parse_args([])
    assert a.domain == "circle_packing"
    assert a.mutator == "agent" and a.analyzer == "agent"
    assert a.generations == 20 and a.batch_size == 2 and a.seed == 42
    assert a.spectral_threshold_mode == "empirical"
    assert a.enable_recombination is False


def test_main_forwards_every_flag_to_esn_run(monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(runner, "_load_domain", lambda name, **kw: f"DOMAIN:{name}")
    monkeypatch.setattr(
        runner, "_build_mutator", lambda kind, domain, model, **kw: f"MUT:{kind}:{model}"
    )
    monkeypatch.setattr(runner, "_build_analyzer", lambda kind, model, **kw: f"AN:{kind}:{model}")
    monkeypatch.setattr(
        runner, "_build_predictor", lambda kind, model, **kw: f"PRED:{kind}:{model}"
    )

    class _Result:
        best_score = 1.0
        best_code = "def solve(): return 0.0"
        generations = 7

    def fake_run(domain, **kw):
        calls.update(kw)
        calls["domain"] = domain
        return _Result()

    monkeypatch.setattr(runner.esn, "run", fake_run)

    runner.main(
        [
            "--domain",
            "tsp",
            "--mutator",
            "llm",
            "--analyzer",
            "llm",
            "--mutation-model",
            "gpt-4o",
            "--analysis-model",
            "gpt-4o-mini",
            "--generations",
            "7",
            "--batch-size",
            "3",
            "--seed",
            "11",
            "--spectral-threshold-mode",
            "hybrid",
            "--enable-recombination",
        ]
    )

    assert calls["domain"] == "DOMAIN:tsp"
    assert calls["mutator"] == "MUT:llm:gpt-4o"
    assert calls["analyzer"] == "AN:llm:gpt-4o-mini"
    assert calls["generations"] == 7
    assert calls["batch_size"] == 3
    assert calls["seed"] == 11
    assert calls["spectral_threshold_mode"] == "hybrid"
    assert calls["enable_recombination"] is True


def test_analyzer_none_yields_no_analyzer():
    assert runner._build_analyzer("none", "any-model") is None
