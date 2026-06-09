# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Flag-driven CLI runner for the bundled ESN examples — run from bash, no Python.

Pick a bundled domain (circle_packing or tsp), the mutator/analyzer tier, models,
and search knobs via flags. Run ``--help`` to see them all.

  # Agentic mutation — key-free Claude subscription ([agent] + [novelty]):
  uv run python examples/run.py --domain circle_packing \
      --mutator agent --analyzer agent \
      --generations 20 --batch-size 2 --seed 42

  # Linear prompt-response mutation — API key ([llm] + [novelty], OPENAI_API_KEY set):
  uv run python examples/run.py --domain circle_packing \
      --mutator llm --analyzer llm \
      --mutation-model gpt-4o --analysis-model gpt-4o-mini \
      --generations 20 --batch-size 4 --seed 42 \
      --spectral-threshold-mode hybrid --enable-recombination
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import esn


def _load_domain(name: str):
    """Build the chosen example DomainSpec (importable regardless of launch cwd)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if name == "circle_packing":
        from circle_packing.domain import create_circle_packing_domain_spec

        return create_circle_packing_domain_spec(timeout_seconds=60)
    if name == "tsp":
        from tsp.domain import create_tsp_domain_spec

        return create_tsp_domain_spec()
    if name == "local_sqli_lab":
        from local_sqli_lab.domain import create_local_sqli_lab_domain_spec

        return create_local_sqli_lab_domain_spec()
    raise ValueError(f"unknown domain {name!r}")


def _build_mutator(kind: str, domain, model: str):
    if kind == "agent":
        return esn.make_agent_mutator(domain, model=model)
    if kind == "llm":
        return esn.make_llm_mutator(domain, model=model)
    raise ValueError(f"unknown mutator {kind!r}")


def _build_analyzer(kind: str, model: str):
    if kind == "none":
        return None  # fitness-only; esn.run warns that novelty is inactive
    if kind == "agent":
        return esn.make_agent_analyzer(model=model)
    if kind == "llm":
        return esn.make_analyzer(model=model)
    raise ValueError(f"unknown analyzer {kind!r}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="esn-run",
        description="Run a bundled ESN example from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domain",
        choices=["circle_packing", "tsp", "local_sqli_lab"],
        default="circle_packing",
    )
    p.add_argument(
        "--mutator",
        choices=["agent", "llm"],
        default="agent",
        help="agent = key-free Claude subscription; llm = API-key single-shot completion",
    )
    p.add_argument(
        "--analyzer",
        choices=["agent", "llm", "none"],
        default="agent",
        help="hypothesis source that activates novelty (N_sp); 'none' = fitness-only",
    )
    p.add_argument("--mutation-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--analysis-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--spectral-threshold-mode",
        choices=["empirical", "mp", "hybrid"],
        default="empirical",
        help="spike-detection threshold for the spectral pipeline",
    )
    p.add_argument(
        "--enable-recombination",
        action="store_true",
        help="let the engine recombine high-performing branches",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("esn").setLevel(logging.INFO)  # show per-generation progress

    domain = _load_domain(args.domain)
    mutator = _build_mutator(args.mutator, domain, args.mutation_model)
    analyzer = _build_analyzer(args.analyzer, args.analysis_model)

    result = esn.run(
        domain,
        mutator=mutator,
        analyzer=analyzer,
        generations=args.generations,
        batch_size=args.batch_size,
        seed=args.seed,
        enable_recombination=args.enable_recombination,
        spectral_threshold_mode=args.spectral_threshold_mode,
    )

    print(
        f"\n=== {args.domain} | best_score={result.best_score:.4f} | generations={result.generations} ==="
    )
    print(result.best_code)


if __name__ == "__main__":
    main()
