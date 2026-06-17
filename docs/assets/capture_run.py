# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Capture a real circle_packing run (novelty on/off) for the value figures.

Drives :class:`ESNEngine` directly so we can snapshot, per generation, every
candidate record + its archive route + the spectral state — data ``esn.run()``
does not surface. Writes ``data/run_{on,off}.json`` next to this script; render
the figures from that data with ``make_value_figures.py``.

This uses the **key-free Claude subscription** (single-shot, Haiku) via the
internal ``_AgentLLMClient`` helper, so it needs the ``[agent]`` + ``[novelty]``
extras on Python 3.11+ for a live spectral signal. The committed ``data/*.json``
is the exact output of one paired run (seed 42); the figures are reproducible
from that data without re-running the LLM.

    uv run --extra agent --extra novelty python docs/assets/capture_run.py on
    uv run --extra agent --extra novelty python docs/assets/capture_run.py off
"""

import json
import logging
import sys
import warnings
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger("esn").setLevel(logging.WARNING)
warnings.simplefilter("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "examples"))

from esn.api import _AgentLLMClient, _build_novelty_stack  # noqa: E402
from esn.core.scorer import compute_gamma_weight  # noqa: E402
from esn.engine.analyzer import LLMAnalyzer  # noqa: E402
from esn.engine.engine import ESNEngine  # noqa: E402
from esn.engine.mutator import LLMMutator  # noqa: E402
from circle_packing.domain import create_circle_packing_domain_spec  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
SEED, GENS, BATCH = 42, 15, 3
MODEL = "claude-haiku-4-5-20251001"
OUT = HERE / "data" / f"run_{MODE}.json"

domain = create_circle_packing_domain_spec(timeout_seconds=60)
mutator = LLMMutator(_AgentLLMClient(MODEL), domain)
if MODE == "on":
    analyzer = LLMAnalyzer(_AgentLLMClient(MODEL))
    knowledge, nc, config = _build_novelty_stack(SEED, "empirical")
else:
    analyzer, knowledge, nc, config = None, None, None, None

engine = ESNEngine(
    domain=domain,
    mutator=mutator,
    predictor=None,
    analyzer=analyzer,
    knowledge=knowledge,
    novelty_computer=nc,
    config=config,
    seed=SEED,
    batch_size=BATCH,
    total_generations=GENS,
    enable_recombination=True,
)
emb = getattr(knowledge, "_embedder", None) if knowledge else None
print(
    f"=== circle_packing | mode={MODE} | embedder={'ON' if emb else 'OFF'} | gens={GENS} batch={BATCH} ==="
)

gen_rows, cand_rows = [], []
for g in range(1, GENS + 1):
    recs = engine.run_batch_generation()
    elite_ids = {c.id for c in engine.elite_archive.get_all()}
    front_ids = {c.id for c in engine.frontier_archive.get_all()}
    fnov = getattr(engine.frontier_archive, "novelty_scores", {}) or {}
    for r in recs:
        route = (
            "failed"
            if not r.success
            else "elite"
            if r.id in elite_ids
            else "frontier"
            if r.id in front_ids
            else "not_retained"
        )
        cand_rows.append(
            {
                "gen": g,
                "id": r.id,
                "parent": r.parent_id,
                "score": float(r.score or 0.0),
                "success": bool(r.success),
                "family": r.family or "",
                "ep": float(r.epistemic_novelty or 0.0),
                "sp": float(r.spectral_novelty or 0.0),
                "route": route,
                "frontier_novelty": float(fnov.get(r.id, 0.0)),
            }
        )
    spikes, erank, persist, gamma = 0, None, 0, 0.0
    ss = getattr(nc, "spectral_state", None) if nc else None
    if ss is not None:
        spikes = int(getattr(ss, "num_spikes", 0))
        er = getattr(ss, "effective_rank", getattr(ss, "erank", None))
        erank = None if er is None else float(er)
        persist = int(getattr(nc, "_spike_persistence", 0))
        if erank is not None:
            gamma = float(
                compute_gamma_weight(
                    erank, config.tau, spikes, persist, config.min_spike_persistence
                )
            )
    fams = {c.family for c in engine.elite_archive.get_all()} | {
        c.family for c in engine.frontier_archive.get_all()
    }
    gen_rows.append(
        {
            "gen": g,
            "best": float(engine._best_score),
            "spikes": spikes,
            "erank": erank,
            "persist": persist,
            "gamma": gamma,
            "elite": engine.elite_archive.size,
            "frontier": engine.frontier_archive.size,
            "families": len(fams),
        }
    )
    print(
        f"[{MODE}] gen {g:2d}: best={engine._best_score:7.4f} spikes={spikes} gamma={gamma:.3f} "
        f"frontier={engine.frontier_archive.size}"
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        {
            "mode": MODE,
            "seed": SEED,
            "gens": GENS,
            "batch": BATCH,
            "model": MODEL,
            "embedder": bool(emb),
            "generations": gen_rows,
            "candidates": cand_rows,
        },
        indent=2,
    )
)
print(f"WROTE {OUT}")
