# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Run the skeleton domain with a REAL, key-free agentic mutator + analyzer.

Passing a real `analyzer` is what activates ESN's epistemic-spectral novelty
(N_sp); without it `esn.run` warns loudly. Both factories authenticate through
your local Claude install (macOS keychain / subscription) -- NO API key. They
need the [agent] extra: `uv sync --extra agent`.

Run it from the repo root with: `uv run python examples/skeleton/run.py`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import esn

# Make the local skeleton package importable regardless of launch cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skeleton.domain import create_domain_spec  # noqa: E402


def main() -> None:
    domain = create_domain_spec()

    mutator = esn.make_agent_mutator(domain, model="claude-haiku-4-5-20251001")
    analyzer = esn.make_agent_analyzer(model="claude-haiku-4-5-20251001")  # activates novelty

    result = esn.run(
        domain,
        generations=3,
        batch_size=2,
        mutator=mutator,
        analyzer=analyzer,
        seed=42,
    )

    print("\n=== knapsack_skeleton x Haiku-agent run ===")
    print(f"best_score:  {result.best_score:.1f}  (greedy seed scores 30.0)")
    print(f"generations: {result.generations}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("esn").setLevel(logging.INFO)
    main()
