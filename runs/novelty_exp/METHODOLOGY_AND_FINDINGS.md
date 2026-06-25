# ESN vs baselines on bias-free circle-packing (4o-mini) — methodology & findings

Consolidated so we stop re-proving the same things. Date: 2026-06-25.

## Task / setup
- Domain: bias-free `circle_packing` — 26 circles in unit square, maximize sum of radii.
  Seed "ring" program scores **1.66023**. AlphaEvolve SOTA ~2.635. A valid 5x5 grid
  (centers 0.1..0.9 spacing 0.2, r=0.1) scores exactly **2.5** (25*0.1 + degenerate 26th).
- Generator: gpt-4o-mini (OpenAI, key = `$OPENAI_API_KEY_ESN`). Mutator/analyzer/predictor
  all via ESN's own `make_llm_mutator` / `make_analyzer` / `make_predictor`.
- Scorer: `examples/circle_packing/domain.py:evaluate_circle_packing_artifact` — VALIDATES
  (shape, finiteness, non-neg radii, bounds eps=1e-6, all-pairs overlap eps=1e-6) THEN scores
  `sum(radii)`. Invalid -> score 0, success=False. **Metric audited sound — no gaming/leak.**
  A reported success score is valid by construction. (`runs/novelty_exp/audit_metric.py`)
- Harness: `runs/novelty_exp/run_specdim.py` (args: `arm seed gens batch`; arm="off" or an int
  = novelty-on with spectral_dim forced). Reports best_score + n_evals + spectral firing stats.

## METHODOLOGY — preconditions for a VALID comparison (each was violated at least once)

1. **Identical prompt across ALL arms.** Every arm must generate via ESN's own LLMMutator
   (same system prompt incl. the "avoid multi-phase optimization, prefer greedy single-pass"
   runtime constraint, same seed). NEVER hand-write a baseline prompt. A hand-written
   "bare objective" baseline let the model use `scipy.optimize` (57/80 candidates), which ESN's
   prompt forbids -> baseline wins on a forbidden strategy. (3rd recurrence of this confound.)

2. **Clean engine — neutralize the PARENT_QUALITY_FLOOR_RATIO gate.** A non-upstream gate
   (require parent score >= 0.85*best) was added during earlier exploration. It CRIPPLES ESN,
   especially novelty-ON (pinned it at the seed, 1.67). Run with `NEUTRALIZE_GATE=1`
   (sets floor=0.0 -> upstream behavior) or on stock engine. Gate is NOT shipped.

3. **Match budget by ACTUAL n_evals, not generations.** `gens=2 batch=80` = 160 evals
   (a 2x leak — both gens run a full batch), not 80. ESN `gens=20 batch=4` yields only ~61-68
   ACTUAL evals (~24% of mutations fail compile/validate and aren't counted). Always report and
   match `n_evals`. Single-shot best-of-N = `gens=1 batch=N` (~N evals after yield).

4. **The score distribution is BIMODAL** (~1.8-2.2 "wrong-spacing grid" vs ~2.5 "exact grid",
   little in between). Compare by **jackpot rate P(reach >=2.4)**, NOT means — means average
   over the bimodality and HIDE the real difference. (This metric error produced a false
   "ESN == best-of-N, indistinguishable" conclusion.)

5. **Unseeded LLM mutator -> huge run-to-run variance.** "seed" only seeds the engine RNG, not
   the LLM. n=2-4 is uninterpretable. Need n>=20/arm for a significant jackpot-rate difference.

## ESTABLISHED FINDINGS (don't re-derive)

- **Spectral novelty was DORMANT on 4o-mini.** spectral_dim=48 -> PCA working dim d=min(48,n_obs)
  tracks the bank size -> gamma=d/n_obs pinned ~1.0 -> always "undersampled" -> 0 spikes,
  N_sp identically 0. So "novelty-ON" was epistemic-only. **FIX: spectral_dim 48 -> 8** (ESNConfig
  default, branch `worktree-fix-spectral-dim`). With dim=8, gamma=8/n drops below the 0.9 gate
  once n>~9 and clears n<30 by ~gen 6; spectral fires (4-21 spike-gens, N_sp live 12-32/run).
  **Mechanism fix is real and shippable.**

- **Engaged spectral provides NO score benefit.** Even firing, novelty-ON (esn_on8) is the
  nominally LOWEST arm on clean/matched grids. Engaging != helping on this task.

- **Single-shot best-of-N rarely reaches 2.5.** Fresh: 0 of ~400 single-shot samples (5 batches
  of 80) hit >=2.3; jackpot grid single-shot = 1/8 reached >=2.4. 4o-mini almost never writes the
  exact r=0.1 grid in one shot (no scipy/optimizer in top candidates — just grid+greedy ~1.9).

- **Iteration reaches 2.5 more often — SUGGESTIVE, not yet significant.** Jackpot grid (n=8, gate
  off, matched-ish budget, iteration with FEWER evals 64 vs 80):
  | arm | P(>=2.4) | mean | evals |
  |---|---|---|---|
  | single-shot (gens=1,b=80) | 1/8 | 2.061 | 80 |
  | iteration (gens=20,b=4)   | 3/8 | 2.145 | 64 |
  Direction is consistent across experiments (fresh single 0/5; matched esn_off 3/4) but
  3/8 vs 1/8 is Fisher p~0.28 — NOT significant at n=8. Needs n>=20-24/arm to confirm.

- **Novelty regime split (earlier, agentic):** Haiku agent full-budget (20-gen) novelty-ON > OFF
  on both seeds (2.498 vs 2.406, n=2, weak); ESN ~= ShinkaEvolve (competitive, no winner).
  Retracted an earlier overclaim table ("fully settled, Shinka loses") — was cherry-picked n=1-2.

## OPEN (the ONE experiment worth running next)
- Powered jackpot-rate test: single-shot vs iteration, n>=24 seeds, clean engine, matched n_evals,
  P(>=2.4). If iteration's edge holds at significance -> thesis (ESN extracts the 2.5 grid that
  single-shot can't) is SUPPORTED. Optionally repeat on a WEAKER model (gpt-3.5-turbo /
  gpt-4.1-nano) where single-shot should be ~0 and iteration's lift cleaner.
- Do NOT re-run: prompt-asymmetry baselines, gated-engine runs, mean-based comparisons,
  spectral-dormancy checks, metric-gaming audits — all settled above.
