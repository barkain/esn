# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Drive the circle-packing example with the single-shot LLM mutator.

The single-shot LLM counterpart to ``examples/run_circle_packing_agent.py``: it
builds a real mutator via :func:`esn.make_llm_mutator` — one OpenAI
chat-completions call per mutation (model ``gpt-4o-mini``).

Auth: reads the ambient ``OPENAI_API_KEY`` from the environment.

Requirements:
  * The ``uv`` CLI must be on PATH: the circle-packing domain compiles each
    candidate in an isolated ``uv run --no-project --with numpy --with scipy``
    subprocess. The first candidate fetches wheels (needs network); later ones
    reuse the cache.
  * The ``[llm]`` extra must be installed: ``uv sync --extra llm``.

Run it from the repo root with:

    uv run python examples/run_circle_packing_llm.py
"""

from __future__ import annotations

import logging

import esn


def _load_domain_factory():
    """Import ``create_circle_packing_domain_spec`` regardless of launch cwd.

    Mirrors ``run_circle_packing_agent.py``: prefer the ``examples`` package
    path (works when run from the repo root), and fall back to putting the
    ``examples`` dir on ``sys.path`` so ``from circle_packing.domain import ...``
    resolves too.
    """
    try:
        from examples.circle_packing.domain import create_circle_packing_domain_spec
    except ImportError:  # pragma: no cover - layout fallback
        import sys
        from pathlib import Path

        examples_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(examples_dir))
        from circle_packing.domain import create_circle_packing_domain_spec

    return create_circle_packing_domain_spec


def main() -> None:
    create_circle_packing_domain_spec = _load_domain_factory()

    # 60s per-candidate sandbox budget (matches the domain's hard constraint).
    domain = create_circle_packing_domain_spec(timeout_seconds=60)

    # Single-shot LLM mutator: one OpenAI chat-completions call per mutation.
    mutator = esn.make_llm_mutator(domain, model="gpt-4o-mini")

    # Short but real: 2 generations x batch 2 = 4 candidate slots, each a real
    # single-shot gpt-4o-mini call.
    result = esn.run(
        domain,
        generations=2,
        batch_size=2,
        mutator=mutator,
        seed=42,
    )

    print("\n=== circle_packing x single-shot-LLM (gpt-4o-mini) run ===")
    print(f"best_score:  {result.best_score:.4f}  (seed floor 1.6602)")
    print(f"generations: {result.generations}")
    print("per-gen history:")
    for row in result.history:
        print(f"  {row}")


if __name__ == "__main__":
    # Surface ESN's per-generation progress on the console.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("esn").setLevel(logging.INFO)
    main()
