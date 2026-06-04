# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""ESN quickstart: a minimal, real, NO-API-KEY agentic run.

Runs ESN on the circle-packing example with a real Claude-agent mutator and a
real analyzer -- the analyzer is what activates ESN's epistemic-spectral novelty
(N_sp); without it `esn.run` warns loudly. Both factories authenticate through
your local Claude install (macOS keychain / subscription), so NO API key is
needed -- only the [agent] extra.

Requirements:
  * The `uv` CLI on PATH (each candidate compiles in an isolated `uv run`
    subprocess; the first one fetches numpy/scipy wheels, later ones reuse cache).
  * The [agent] extra installed: `uv sync --extra agent`.

Run it from the repo root with: `uv run python examples/quickstart.py`.
"""

from __future__ import annotations

import logging

import esn


def main() -> None:
    try:
        from examples.circle_packing.domain import create_circle_packing_domain_spec
    except ImportError:  # pragma: no cover - layout fallback
        from examples.circle_packing import create_circle_packing_domain_spec

    domain = create_circle_packing_domain_spec()

    mutator = esn.make_agent_mutator(domain, model="claude-haiku-4-5-20251001")
    analyzer = esn.make_agent_analyzer(model="claude-haiku-4-5-20251001")  # activates novelty

    result = esn.run(
        domain,
        generations=2,
        batch_size=2,
        mutator=mutator,
        analyzer=analyzer,
        seed=42,
    )

    print(f"Generations run: {result.generations}")
    print(f"Best score: {result.best_score:.4f}  (seed floor 1.6602)")


if __name__ == "__main__":
    # Surface ESN's per-generation progress on the console.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("esn").setLevel(logging.INFO)
    main()
