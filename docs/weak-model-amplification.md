# Amplifying a weak model's intelligence

> **TL;DR.** A small/cheap model (here `gpt-4o-mini`) gets *stuck* on hard search
> problems — it rebuilds the same kind of solution and cannot find genuinely new,
> better ones. We tested five ways to unlock it. Four do nothing. The one that
> works is **objective-grounded feedback**: tell the model its real score and *why*
> it fell short (computed from the evaluator), and let it revise. On a clean,
> hardened metric, *when it succeeds* a feedback-driven escape closes ~**45%**
> (conditional median) to ~**75%** (peak) of the gap to a much stronger model
> (`gpt-4o`), at **matched compute** — and it succeeds on ~**7%** of attempts versus
> ~**0%** without, on a problem the weak model otherwise makes *zero* progress on. This document records what we tested, the metric we use to judge
> "did this actually add intelligence (vs just more compute)," and the honest limits.

This is an empirical findings note. It complements [how-it-works.md](how-it-works.md)
(the engine's architecture) by reporting **what actually moves the needle** for a
weak model, and qualifying claims about the spectral-novelty signal.

---

## 1. The framing: amplifier, not solver

Treat the weak model as a *low-capability problem-solver* and the algorithmic
technique as a **cognitive scaffold**. The goal is **not** to solve every problem
(reach the global optimum); it is to show the scaffold makes the weak model
**measurably better than it is on its own** — and to quantify *how much* better
counts as a real improvement.

The central trap to avoid: **more compute is not more intelligence.** Giving the
model more tries (best-of-N sampling, or "just retry") is more shots on goal, not a
smarter solver. A technique only earns the label *intelligence amplification* when
it does **better with the same compute budget**.

---

## 2. The metric: a two-stage test

**Stage 1 — Capability gate (binary).** The technique must *statistically beat the
matched-compute control* — the same number of model calls/tokens, the same retry
structure, the same evaluator access, the same selection rule — ideally **crossing a
ceiling the weak model provably cannot cross by sampling alone**. If it fails this,
it is compute orchestration, not added intelligence. Stop here.

**Stage 2 — Magnitude (only if the gate passes).** Reported on two axes, three
numbers each (the *typical* number is primary; *peak* alone is just tail luck):

| | definition | reading |
|---|---|---|
| **Tier lift** | `(technique − weak) / (strong − weak)` | % of the **weak→strong** gap closed — the amplification axis |
| **Task progress** | `(technique − weak) / (target − weak)` | distance toward **solved** — keeps us honest when even the strong model is far from solving |

For each axis: **typical** (median over matched-budget runs), **peak** (best
observed), **reliability** (fraction of runs that cross the bar).

**Viability convention** (a convention, not a law of nature):

- **Viable intelligence improvement** — passes the capability gate **and** closes
  **≥ ~25–30%** of the weak→strong gap in *typical* terms.
- **Promising but unreliable** — crosses the barrier (peak/existence is real) but
  typical lift ≈ 0.
- **Not viable** — does not beat matched compute.

---

## 3. What we tested — the ladder

Circle-packing (pack 26 circles in the unit square; maximize **sum of radii**).
Metric: an order-free LP radius allocator + hardening (score ≥ 2.30 **and** all 26
circles non-zero **and** non-ring morphology **and** beats the parent 2.229). The
weak model alone **never** clears this bar.

| technique given to the weak model | hardened-escape rate | verdict |
|---|---|---|
| no steer | **0 / 192** | copies the incumbent |
| generic "you converged, diverge" | **0 / 192** | explores but unproductively |
| the engine's spectral-novelty guidance (as injected) | **0 / 192** | does not author a useful direction |
| **self-diagnosis** (model names its own failure + a fix) | **0 / 96** | confident but *misaligned* (see §4) |
| just-retry (second attempt, generic) | **3 / 192** (~1.6%) | more compute, not intelligence |
| a **human expert hint** ("ring plateaued → categorically different arrangement") | **18 / 192** (~9%) | reference ceiling |
| **objective-grounded feedback** | **~7%** (13/192 pooled), `p = 0.009` vs just-retry | **the lever** ✅ |

Among the automated (non-human) interventions, only objective-grounded feedback
beats the matched-compute control, and it reaches the same order of escape rate as a
hand-written expert hint — **with no human in the loop.**

---

## 4. Why self-diagnosis fails and feedback works

When asked to diagnose itself, the weak model's advice was **concrete but aimed at
the wrong target**. It proposed *"hexagonal close-packing," "triangular lattice,"
"Voronoi tessellation"* — the textbook answer for cramming the **most circles in**.
But the objective here is **sum of radii** (a few big circles + fill), a *different*
problem. It pattern-matched the *famous* version of the task, committed to it, and
**never once beat the starting solution** (best 2.229 = exact tie, 0 hardened).

The lesson: **a specific *wrong* direction can fail just as badly as no direction**
(self-diagnosis and the generic nudge both score 0); committing to a plausible-but-
wrong family buys nothing, whereas the open-ended human hint leaves room to stumble
onto an arrangement that works. The fix has to come from **what actually scores**, not the model's intuition
about the famous problem.

That is exactly what feedback does. The feedback is computed from the *real*
evaluator — e.g. *"you scored 2.23; 14 of 26 circles are wasted at near-zero radius;
each radius is capped by the nearest neighbour and the wall, so spread the centres so
more circles can grow"* — and it **never names a strategy** ("grid"/"hex"). Grounded
in the objective, it cannot pattern-match to the wrong problem; it reports the truth.

---

## 5. Two domains, and the intelligence-lift table

| domain (barrier type) | weak floor (matched sampling) | strong ceiling | technique | tier lift (conditional / peak) | reliability | gate | verdict |
|---|---|---|---|---|---|---|---|
| **circle-packing** (*direction* problem) | 2.229 (best-of-60, never exceeds) | 2.610 (`gpt-4o` best-of-60) | objective-grounded feedback (1 round) | **45% / 75%** | ~7% of attempts escape | ✅ `p=0.009` | **viable** |
| **SQLi lab** (*precision* problem) | 520 (0/180 ever above) | 750 (`gpt-4o` best-of-180) | feedback via the full **multi-round** engine loop | **17% / 52%** | 6/10 search runs cross 520 | ✅ crosses the 520 cap | **promising** |

The **tier-lift numbers are *conditional on success*** (the median/peak quality of an
escape, as a fraction of the weak→strong gap) — *not* a median over all attempts,
which is ~0% because most individual attempts don't escape. The **reliability**
column carries that "how often." Read them together: a *viable* result needs both a
meaningful conditional lift **and** non-trivial reliability above the matched-compute
control.

A single feedback **round** suffices when the barrier is *which direction to go*
(circle-packing). When the barrier is *precise execution* (SQLi: build an exact
blind-extraction template), one round is not enough — the weak model needs the
feedback applied **repeatedly** over many generations. The engine's iterative loop
does reach 640 (6/10 seeds) where single-shot sampling is hard-capped at 520.

So, across these two domains: **the number of feedback rounds needed appears to
scale with the barrier type** — one for
direction, many for precision.

---

## 6. The practical recipe

1. Let the weak model make an attempt.
2. **Score it with the real evaluator**, and turn the evaluator's own diagnostics
   into a *positive, objective-grounded* directive — *what fell short and which
   measured quantity to move* — **without** prescribing a strategy.
3. Feed that back and let it revise. **Repeat** until the score stalls; the harder
   the precision barrier, the more rounds you need.

What to avoid (all measured to do nothing): asking the model to "be novel," asking
it to diagnose and direct *itself*, or relying on a domain-agnostic novelty signal to
supply the direction. Direction must come from the objective, not the model's priors.

> **Note on the spectral-novelty signal.** ESN's spectral guidance is a real and
> useful *measurement* (it detects when the hypothesis population has spread or
> stagnated), but in our tests the guidance text *as injected* did not by itself
> unlock structural escape (0/192): it **did not author the missing domain-specific
> direction** — at most it can surface/weight directions already present in the
> hypotheses or observations. This does *not* rule out value from archive/frontier
> effects or a redesigned guidance policy; it does say the actionable direction, in
> our tests, came from objective feedback rather than the signal. Best used as a
> *trigger/diagnostic*. **Scope:** this is *only* about the guidance-*text*-as-prompt-steer
> mechanism. It is **not** a claim about novelty-guided *selection* (`N_sp` ranking the
> frontier and choosing parents) — ESN's core premise, which steers the search and is
> unaffected by this finding. The two coexist: novelty *selection* earns its place; the
> guidance *text*, pasted into the prompt, doesn't author a domain direction. See
> [how-it-works.md](how-it-works.md) for the mechanism.

---

## 7. Confounds we hit (so you don't repeat them)

- **Scoring artifacts dominate.** A naive radius allocator inflated an apparent
  "structural leap" by ~0.68 (most of it); the honest gain is ~+8%. **Always**
  score with an order-free allocator and require all circles non-zero (the validator
  permits zero-radius circles, which games the sum).
- **Strong models memorize.** `gpt-4o` reproduces the *classical* circle-packing
  optimum (~2.61 ≈ known best) within a handful of samples — consistent with recall
  or a memorized/classical prior, not clean evidence of search. Use a
  **custom, contamination-free** task (the SQLi lab) to measure capability honestly.
- **Validate the measuring stick.** Every escape harness must reproduce a known
  baseline (here: the human-hint arm at ~7–9%) *before* its numbers are trusted. One
  teammate harness silently truncated outputs (`max_tokens=600`) and gave the known-
  good human arm 0% — caught only by the validation arm.
- **Don't infer progress from a buffered file.** A block-buffered results file read
  as "0 trials" mid-run and a watchdog killed healthy runs; open progress files
  line-buffered.

---

## 8. Scope and limits

- **Reliably shown on one domain** (circle-packing), **promising on a second** (SQLi).
  A general "viable intelligence improvement" claim needs replication across more
  qualitatively different domains.
- Counts are modest (e.g. 13 vs 3 escapes); results are significant but not
  large-sample. The human-hint ceiling is noisy (~3.6–9.4% across runs).
- The strong model is used **only** as a diagnostic ceiling, never as the proposed
  solution.
- Honest caveat to the amplifier story: on the contamination-free SQLi task, a
  *stronger model merely sampled* (750) still beat *weak + algorithm* (640). The
  technique reliably amplifies the weak model **past what it can do alone**; it does
  not make it beat a stronger model outright.

---

## 9. Reproducing

The experiment harnesses live in [`assets/experiments/`](assets/experiments/):

| file | what it measures |
|---|---|
| `cp_a3_run.py` | the ladder on circle-packing (no-steer / generic / spectral / human-hint) |
| `cp_self_exp.py` | self-diagnosis vs human hint (validated harness) |
| `cp_feedback_exp.py` | objective-grounded feedback vs just-retry (circle-packing) |
| `sqli_feedback_exp.py` | feedback-learning on the contamination-free SQLi lab (paired design) |

Each prints a results table and validates against a known baseline before trusting
its numbers. Set `OPENAI_API_KEY` to a key with `gpt-4o-mini` access.
