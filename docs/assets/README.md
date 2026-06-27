# ESN figures

Figures rendered from **real `circle_packing` run data** — not mock-ups. The
README hero is a *performance* figure; the other two illustrate **mechanism**
(how the search routes candidates), where controlled multi-seed comparisons did
*not* show a novelty advantage on this benchmark — so those two are not a
"novelty wins" claim.

| Figure | Shows |
| --- | --- |
| `evolution-climbs-to-sota.png` (README hero) | The two real heavy-evolution runs (gpt-4o-mini, fixed prompt, seeds 42 & 43) climbing in discrete steps from the scipy seed (~2.595) past the best-of-N sampling plateau (~2.614) up to ≈ the AlphaEvolve SOTA (2.635). A budget-dependent *performance* result (heavy arm n=2; see the study). |
| `frontier-survival-circle-packing.png` | Every candidate by generation/score, colored by archive route (Claude Haiku, seed 42). On this run the best (2.06) is the child of a **below-best** (1.75) candidate the novelty frontier kept alive — an illustration of the routing rule, not a performance result. |
| `spectral-gate-circle-packing.png` | The spectral mixing weight γ stays at zero until spikes persist ≥ 3 generations — spectral steering is conservative by construction. |

## Reproduce

The hero renders from the committed heavy-run trajectories
(`runs/novelty_exp/results_heavy.jsonl`) with no LLM:

```bash
uv run --with matplotlib python docs/assets/make_hero_figure.py
```

The mechanism figures render from the committed `data/run_{on,off}.json`, no LLM:

```bash
uv run --with matplotlib --extra novelty python docs/assets/make_value_figures.py
```

To re-capture the underlying data (needs the `[agent]` + `[novelty]` extras and a
Claude subscription; runs the live LLM):

```bash
uv run --extra agent --extra novelty python docs/assets/capture_run.py on
uv run --extra agent --extra novelty python docs/assets/capture_run.py off
```

`capture_run.py` drives `ESNEngine` directly to snapshot per-generation candidate
records, archive routing, and spectral state — data `esn.run()` does not surface.
