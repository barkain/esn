# Mechanism figures

Two figures illustrating *how* the ESN search routes candidates, rendered from
**real `circle_packing` run data** (Claude Haiku, seed 42) — not mock-ups. These
depict **mechanism**, not performance: controlled multi-seed comparisons did not
show a novelty advantage on this benchmark, so nothing here is a "novelty wins"
claim.

| Figure | Shows |
| --- | --- |
| `frontier-survival-circle-packing.png` | Every candidate by generation/score, colored by archive route. On this run the best (2.06) is the child of a **below-best** (1.75) candidate the novelty frontier kept alive — an illustration of the routing rule, not a performance result. |
| `spectral-gate-circle-packing.png` | The spectral mixing weight γ stays at zero until spikes persist ≥ 3 generations — spectral steering is conservative by construction. |

## Reproduce

The figures render from the committed `data/run_{on,off}.json` with no LLM:

```bash
uv run --extra novelty python docs/assets/make_value_figures.py
```

To re-capture the underlying data (needs the `[agent]` + `[novelty]` extras and a
Claude subscription; runs the live LLM):

```bash
uv run --extra agent --extra novelty python docs/assets/capture_run.py on
uv run --extra agent --extra novelty python docs/assets/capture_run.py off
```

`capture_run.py` drives `ESNEngine` directly to snapshot per-generation candidate
records, archive routing, and spectral state — data `esn.run()` does not surface.
