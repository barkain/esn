# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Drive the circle-packing example with the Claude-agent (Haiku) mutator.

This is the agentic counterpart to ``examples/quickstart.py``: it builds a real
agentic mutator with :func:`esn.make_agent_mutator` (model
``claude-haiku-4-5-20251001``, no research tools) and runs a short but real
evolution.

Auth: the Claude Agent SDK authenticates via the local Claude install
(macOS keychain / subscription). It does NOT require ``ANTHROPIC_API_KEY`` in
the environment, so we neither read nor demand it here.

Requirements:
  * The ``uv`` CLI must be on PATH: the circle-packing domain compiles each
    candidate in an isolated ``uv run --no-project --with numpy --with scipy``
    subprocess. The first candidate fetches wheels (needs network); later ones
    reuse the cache.
  * The ``[agent]`` extra must be installed: ``uv sync --extra agent``.

Run it from the repo root with:

    uv run python examples/run_circle_packing_agent.py
"""

from __future__ import annotations

import logging

import esn


def _load_domain_factory():
    """Import ``create_circle_packing_domain_spec`` regardless of launch cwd.

    Mirrors ``quickstart.py``: prefer the ``examples`` package path (works when
    run from the repo root), and fall back to putting the ``examples`` dir on
    ``sys.path`` so ``from circle_packing.domain import ...`` resolves too.
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

    # Haiku agentic mutator: no research tools ("none"), 300s per SDK call so a
    # wedged call is trapped and recorded as a failed candidate rather than
    # hanging the whole run.
    mutator = esn.make_agent_mutator(
        domain,
        model="claude-haiku-4-5-20251001",
        mutator_tools="none",
        call_timeout_seconds=300,
    )

    # Short but real: 3 generations x batch 2 = 6 candidate slots, each a real
    # Haiku SDK call.
    result = esn.run(
        domain,
        generations=3,
        batch_size=2,
        mutator=mutator,
        seed=42,
    )

    print("\n=== circle_packing x Haiku-agent run ===")
    print(f"best_score:  {result.best_score:.4f}  (seed floor 1.6602)")
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
