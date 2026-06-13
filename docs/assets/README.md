# Value figures

Three figures depicting what the ESN search actually does, rendered from **real
`circle_packing` run data** (Claude Haiku, seed 42) — not mock-ups.

| Figure | Shows |
| --- | --- |
| `frontier-survival-circle-packing.png` | Every candidate by generation/score, colored by archive route. The run's best (2.06) descends from a **below-best** (1.75) frontier survivor a greedy loop would discard. |
| `spectral-gate-circle-packing.png` | The spectral mixing weight γ stays at zero until spikes persist ≥ 3 generations — spectral steering is conservative by construction. |
| `novelty-on-vs-control-circle-packing.png` | Novelty-on vs a fitness-only control (same domain, seed, mutator, budget). A single illustrative paired trace, **not a benchmark**. |

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
