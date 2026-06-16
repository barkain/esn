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

  # Diff (SEARCH/REPLACE) edits, several seeds, raised token cap, injected task hint:
  uv run python examples/run.py --domain circle_packing \
      --mutator diff --analyzer llm \
      --mutation-model gpt-4o-mini --analysis-model gpt-4o-mini \
      --generations 30 --batch-size 5 --seeds 42,7,123 \
      --eval-timeout 120 --max-tokens 16384 \
      --task-prompt "Pack 26 circles in a unit square; maximize sum of radii; use scipy SLSQP."

  # circle_packing with feasibility repair (changes the evaluator — see docs/mutators.md):
  uv run python examples/run.py --domain circle_packing --mutator llm --analyzer none \
      --mutation-model gpt-4o-mini --repair --generations 40 --batch-size 4 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import esn


def _load_domain(name: str, *, eval_timeout: int = 60, repair: bool = False, task_prompt=None):
    """Build the chosen example DomainSpec (importable regardless of launch cwd)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if name == "circle_packing":
        from circle_packing.domain import create_circle_packing_domain_spec

        domain = create_circle_packing_domain_spec(timeout_seconds=eval_timeout, repair=repair)
    elif name == "tsp":
        if repair:
            raise ValueError("--repair is only supported for the circle_packing domain")
        from tsp.domain import create_tsp_domain_spec

        domain = create_tsp_domain_spec()
    elif name == "local_sqli_lab":
        if repair:
            raise ValueError("--repair is only supported for the circle_packing domain")
        from local_sqli_lab.domain import create_local_sqli_lab_domain_spec

        domain = create_local_sqli_lab_domain_spec()
    else:
        raise ValueError(f"unknown domain {name!r}")
    if task_prompt:
        # Override the task description the mutator sees (e.g. inject expert hints).
        try:
            domain.description = task_prompt
        except Exception:  # noqa: BLE001 — frozen models need object.__setattr__
            object.__setattr__(domain, "description", task_prompt)
    return domain


def _build_mutator(kind: str, domain, model: str, *, max_tokens=None):
    if kind == "agent":
        return esn.make_agent_mutator(domain, model=model)
    if kind == "llm":
        return esn.make_llm_mutator(domain, model=model, max_tokens=max_tokens)
    if kind == "diff":
        return esn.make_llm_mutator(
            domain, model=model, mutator_policy="diff", max_tokens=max_tokens
        )
    raise ValueError(f"unknown mutator {kind!r}")


def _build_analyzer(kind: str, model: str):
    if kind == "none":
        return None  # fitness-only; esn.run warns that novelty is inactive
    if kind == "agent":
        return esn.make_agent_analyzer(model=model)
    if kind == "llm":
        return esn.make_analyzer(model=model)
    raise ValueError(f"unknown analyzer {kind!r}")


def _build_predictor(kind: str, model: str):
    """Predictor for the Task-1 prediction-surprise novelty term.

    Tied to the analyzer tier: a 'none' analyzer means novelty is off, so no
    predictor either. Otherwise a predictor matching the analysis model is wired
    (this is on by default — the reference v3 benchmark runs with it, and without
    it the epistemic prediction-surprise signal that feeds selection never forms).
    """
    if kind == "agent":
        return esn.make_agent_predictor(model=model)
    if kind == "llm":
        return esn.make_predictor(model=model)
    return None  # 'none' analyzer → novelty inactive → no predictor


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
        choices=["agent", "llm", "diff"],
        default="agent",
        help="agent = key-free Claude subscription; llm = API single-shot full-rewrite; "
        "diff = API SEARCH/REPLACE incremental edits",
    )
    p.add_argument(
        "--analyzer",
        choices=["agent", "llm", "none"],
        default="agent",
        help="hypothesis source that activates novelty (N_sp); 'none' = fitness-only",
    )
    p.add_argument("--mutation-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--analysis-model", default="claude-haiku-4-5-20251001")
    p.add_argument(
        "--no-predictor",
        action="store_true",
        help="disable the Task-1 predictor (prediction-surprise novelty term); "
        "by default a predictor matching --analysis-model is wired whenever the "
        "analyzer activates novelty",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="enable the domain-agnostic ParameterTuner: evaluator-guided "
        "pattern search over a candidate's float literals (matures promising/"
        "novel candidates). Spends extra evaluator calls — count them in any "
        "budget-matched comparison.",
    )
    p.add_argument(
        "--tune-evals",
        type=int,
        default=16,
        help="max evaluator calls the tuner spends per candidate",
    )
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seeds to run sequentially (overrides --seed), e.g. 42,7,123",
    )
    p.add_argument(
        "--eval-timeout",
        type=int,
        default=60,
        help="per-candidate evaluation timeout in seconds (circle_packing)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="max completion tokens per LLM mutation (raise for long programs; "
        "llm/diff mutators only)",
    )
    p.add_argument(
        "--repair",
        action="store_true",
        help="circle_packing: cheaply project invalid packings to feasibility before scoring",
    )
    p.add_argument(
        "--task-prompt",
        default=None,
        help="override the domain task description the mutator sees (e.g. inject expert hints)",
    )
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

    domain = _load_domain(
        args.domain,
        eval_timeout=args.eval_timeout,
        repair=args.repair,
        task_prompt=args.task_prompt,
    )
    mutator = _build_mutator(args.mutator, domain, args.mutation_model, max_tokens=args.max_tokens)
    analyzer = _build_analyzer(args.analyzer, args.analysis_model)
    predictor = None if args.no_predictor else _build_predictor(args.analyzer, args.analysis_model)
    tuner = None
    if args.tune:
        from esn.engine.tuner import ParameterTuner

        tuner = ParameterTuner(max_evals=args.tune_evals)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for seed in seeds:
        result = esn.run(
            domain,
            mutator=mutator,
            analyzer=analyzer,
            predictor=predictor,
            tuner=tuner,
            generations=args.generations,
            batch_size=args.batch_size,
            seed=seed,
            enable_recombination=args.enable_recombination,
            spectral_threshold_mode=args.spectral_threshold_mode,
        )
        results.append((seed, result))
        print(
            f"\n=== {args.domain} | seed={seed} | best_score={result.best_score:.4f} "
            f"| generations={result.generations} ==="
        )
        if len(seeds) == 1:
            print(result.best_code)

    if len(seeds) > 1:
        scores = [r.best_score for _, r in results]
        mean = sum(scores) / len(scores)
        print(
            f"\n=== {args.domain} | {len(seeds)} seeds | "
            f"best_scores={[round(s, 4) for s in scores]} | mean={mean:.4f} ==="
        )


if __name__ == "__main__":
    main()
