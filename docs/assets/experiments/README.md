# Weak-model amplification experiments

Reference harnesses for [../../weak-model-amplification.md](../../weak-model-amplification.md).
These are **research scripts**, not part of the library: they write artifacts/logs
to `/tmp` and assume `OPENAI_API_KEY` is set to a key with `gpt-4o-mini` (and, for
the strong-ceiling arms, `gpt-4o`) access. Run from the repo root with the project
venv, e.g. `OPENAI_API_KEY=$KEY N=96 .venv/bin/python docs/assets/experiments/cp_feedback_exp.py`.

| file | measures | key control |
|---|---|---|
| `cp_a3_run.py` | the ladder on circle-packing: no-steer / generic-diverge / real spectral guidance / human hint | reads `a3_steers.json` (real engine guidance text); human-hint arm validates the harness (~9%) |
| `cp_self_exp.py` | self-diagnosis (model authors its own steer) vs human hint | HUMAN arm must reproduce ~7–9% or the run is discarded |
| `cp_feedback_exp.py` | objective-grounded feedback vs just-retry (PLACEBO), circle-packing | PLACEBO = same 2-call budget, generic feedback → isolates "intelligence" from "more compute" |
| `sqli_feedback_exp.py` | feedback-learning on the contamination-free SQLi lab | **paired** design (both arms branch from the same round-1 attempt); feedback grounded in real evaluator residuals, no strategy leak |

Every escape metric is hardened (order-free LP allocator + all-26-circles-non-zero
+ non-ring, beating the parent) so scoring artifacts can't masquerade as discovery.
The discipline that matters most: **a harness is only trusted once it reproduces a
known baseline** (the human-hint arm). `a3_steers.json` is the serialized real
`spectral_guidance` dict captured from the engine, used verbatim as a prompt block.
