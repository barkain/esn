# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Drive the TSP example with the Claude-agent (Haiku) mutator.

Agentic counterpart to the offline TSP run: builds a real Claude Agent SDK
mutator via :func:`esn.make_agent_mutator` (model ``claude-haiku-4-5-20251001``,
no research tools) and runs a short but real evolution.

Auth: the Claude Agent SDK authenticates via the local Claude install
(macOS keychain / subscription). It does NOT require ``ANTHROPIC_API_KEY`` in
the environment, so we neither read nor demand it here.

Requirements:
  * The ``uv`` CLI must be on PATH: the TSP domain compiles each candidate in
    an isolated ``uv run`` subprocess. TSP candidates are stdlib-only, so no
    wheels are fetched.
  * The ``[agent]`` extra must be installed: ``uv sync --extra agent``.

Run it from the repo root with:

    uv run python examples/run_tsp_agent.py
"""

from __future__ import annotations

import logging

import esn


def _load_domain_factory():
    """Import ``create_tsp_domain_spec`` regardless of launch cwd.

    Mirrors ``run_circle_packing_agent.py``: prefer the ``examples`` package
    path (works when run from the repo root), and fall back to putting the
    ``examples`` dir on ``sys.path`` so ``from tsp.domain import ...`` resolves.
    """
    try:
        from examples.tsp.domain import create_tsp_domain_spec
    except ImportError:  # pragma: no cover - layout fallback
        import sys
        from pathlib import Path

        examples_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(examples_dir))
        from tsp.domain import create_tsp_domain_spec

    return create_tsp_domain_spec


def main() -> None:
    create_tsp_domain_spec = _load_domain_factory()

    domain = create_tsp_domain_spec()

    mutator = esn.make_agent_mutator(domain, model="claude-haiku-4-5-20251001")

    # Short but real: 2 generations x batch 2 = 4 candidate slots, each a real
    # Haiku SDK call.
    result = esn.run(
        domain,
        generations=2,
        batch_size=2,
        mutator=mutator,
        seed=42,
    )

    print("\n=== tsp x Haiku-agent run ===")
    print(f"best_score:  {result.best_score}")
    print(f"generations: {result.generations}")
    print("per-gen history:")
    for row in result.history:
        print(f"  {row}")


if __name__ == "__main__":
    # Surface ESN's per-generation progress; quiet the noisier SDK transport logs.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("esn").setLevel(logging.INFO)
    main()
