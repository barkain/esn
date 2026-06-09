# How ESN works

ESN is **LLM-driven evolutionary search with a memory**. An LLM proposes
mutations of a candidate program, each candidate is compiled and scored, and the
best ones seed the next round. That part is the familiar "LLM in a loop." What
makes ESN different is that it keeps a running, structured memory of *what the
search has already learned* — a bank of small causal **hypotheses** extracted
from every evaluated candidate — and it uses a spectral analysis of that memory
to ask, for each new candidate: *how structurally unlike everything understood so
far is this?* That signal (`N_sp`) is mixed into how the search picks parents and
what it keeps. The failure mode it fixes is the one every naive loop hits:
collapse. A greedy "keep the highest score" loop quickly converges onto a single
idea and then spends expensive LLM calls re-deriving variations of it. ESN
deliberately routes some of its budget toward candidates that *teach it
something new*, without letting fitness crash — exploration that stays viable and
exploitation that resists stagnation.

This page explains the mechanism end to end — the generation loop, the
hypothesis memory, the spectral math that produces `N_sp`, how candidates are
selected and archived, and the supporting machinery — for someone deciding
whether to adopt ESN and wanting to understand *why* it should beat the naive
loop. To wire your own problem in, see
[connecting-a-problem.md](connecting-a-problem.md); to pick a mutator, see
[mutators.md](mutators.md).

A note on accuracy up front: the [README](../README.md) frames selection as an
"epsilon-band Pareto" rule, and that is the right *intuition* for what the search
reaches for. The implementation realizes that intuition through several
cooperating mechanisms rather than one literal Pareto band; where the naming runs
ahead of the code is spelled out in
[Selection](#5-selection-the-viability-bar-and-where-novelty-actually-bites)
below.

---

## 1. The big idea in one paragraph

Maintain a memory of structure. Every evaluated candidate is distilled by an
**analyzer** into a few confidence-weighted hypotheses ("recursion helped here,"
"this constraint binds"). Once per generation ESN treats that memory as a matrix,
decomposes it, and uses random-matrix theory to separate *real emergent
structure* (spectral "spikes") from noise. A candidate's **spectral novelty**
`N_sp` is then the fraction of what it touched in memory — the certainty-weighted
signature of the hypotheses it engaged — that lies *outside* the known
structured subspace; high when it explores genuinely new territory, near zero
when it rehashes what is already understood. ESN blends `N_sp` with an
**epistemic novelty** `N_ep` (how much a candidate revised the memory's beliefs),
and uses the blend to steer which parents get picked and which non-best
candidates are kept in the archive. Fitness still gates everything — a candidate
must run and score well to matter — but among viable options the search is
biased toward the informative ones.

---

## 2. The generation loop, end to end

The engine ([`ESNEngine`](../src/esn/engine/engine.py)) runs one generation at a
time. With a batch size above 1 it mutates *k* candidates per generation in three
phases; the single-candidate path mirrors the same steps inline.

```
              ┌──────────────────────────────────────────────────────────┐
   seed code ─▶│ evaluate seed once → initial best, branch root          │
              └──────────────────────────────────────────────────────────┘
                                      │
     ┌────────────────────────────────▼───────────────────────────────────┐
     │  PLAN (sequential)                                                  │
     │   • pick search mode (EXPLORE / EXPLOIT / REPAIR / …)               │
     │   • pick the per-generation batch size (budget controller)          │
     │   • assign each slot a (parents, style) — slot 0 is always refine   │
     └────────────────────────────────┬───────────────────────────────────┘
                                      │
     ┌────────────────────────────────▼───────────────────────────────────┐
     │  RUN each slot (parallel, side-effect free)                         │
     │   predict ─▶ mutate ─▶ validate ─▶ compile ─▶ evaluate ─▶ improve   │
     │                                              └─▶ analyze (hypotheses)│
     └────────────────────────────────┬───────────────────────────────────┘
                                      │
     ┌────────────────────────────────▼───────────────────────────────────┐
     │  COMMIT (sequential)                                                │
     │   • operator credit (UCB) • epistemic + spectral novelty            │
     │   • archive successes (elite vs frontier)  • branch lineage         │
     │   • update best (deadband) • stagnation/temperature                 │
     │   • end-of-generation: spectral re-analysis, dedup, pruning         │
     └────────────────────────────────┬───────────────────────────────────┘
                                      │
                            repeat next generation
```

A few design choices in this loop matter for adopters:

- **The seed is evaluated once.** If it compiles and scores, it becomes the
  initial best and the root of the branch tree. If it fails, the search proceeds
  anyway from an unscored seed (best stays `0.0`) — it degrades, it doesn't crash.
- **Phase 2 is pure.** Each slot's `mutate→compile→evaluate→improve→analyze` runs
  with no shared-state writes, returning a self-contained outcome. That is what
  makes batched candidates safe to run on threads. (The in-process
  `PythonSandboxCompiler` is the exception: its timeout uses `SIGALRM`, which only
  works on the main thread, so it forces sequential execution.)
- **All state mutation happens in Phase 3**, candidate by candidate, then once
  per generation. Best-score promotion, stagnation, temperature, the spectral
  re-analysis, memory maintenance, and pruning all happen here exactly once.
- **Success gates almost everything.** A `success=False` candidate is recorded
  and still credits its operator (with a penalty), but it never becomes the best,
  never enters an archive, and never feeds the embedding-based branch geometry.
- **Repeated failure triggers a snap-back.** After two consecutive all-failed
  generations the engine short-circuits planning entirely — mode forced to
  EXPLOIT, a single `refine` of the current best — to claw back to a working
  program before resuming exploration.

There is also an **adaptive batch-budget controller**
([`batch_budget.py`](../src/esn/engine/batch_budget.py)) that sizes each
generation's batch. After a short warm-up it freely *shrinks* the batch, but
*expanding* above the nominal size must clear six simultaneous gates (enough
distinct branches and families, low duplication and collapse rates, recent
improvement, and pace within budget). In practice it mostly conserves budget,
spending more only when the search is demonstrably productive.

---

## 3. Hypotheses: the memory `N_sp` is measured against

`N_sp` is meaningless without something to be novel *against*. That something is
the **hypothesis bank** — the persistent memory of what the search has learned.

After a candidate succeeds, the **analyzer** turns it into a small structured
record: binary **evidence** about existing hypotheses (`{hypothesis_id,
evidence∈{0,1}}`) and up to three **new hypotheses** (`{text, concepts}`). Each
hypothesis ([`HypothesisRecord`](../src/esn/core/spectral_models.py)) carries a
text, a `confidence∈[0,1]`, an observation count `n_obs`, an **embedding**, a set
of concept tags, and a lifecycle `status` (active / retired / archived).

The lifecycle keeps the memory clean so the spectral pipeline runs over a
low-redundancy matrix:

- **Update (Beta-Bernoulli).** Evidence revises a hypothesis's confidence:
  `c' = (c·n + e)/(n+1)`, with a revision magnitude `δ = |e − c|/(n+1)`. That `δ`
  is exactly what feeds epistemic novelty (§6).
- **Admission gate.** A proposed new hypothesis is *rejected only if* it is both
  cosine-similar (≥ 0.88) **and** tag-overlapping (Jaccard ≥ 0.3) to an existing
  active hypothesis — i.e. a near-duplicate. (With no embedder, embeddings are
  zero vectors and everything is auto-admitted; dedup effectively turns off.)
- **Retirement.** A hypothesis retires when it is well-tested but unconvincing
  (`confidence < 0.1` with `n_obs ≥ 10`) or simply never re-tested within a TTL.
- **Dedup.** Maintenance periodically merges near-duplicate hypotheses by pooling
  their evidence (Bayesian pseudo-counts), archiving the absorbed ones.

A crucial detail: novelty is measured against the memory **before** the current
candidate's own evidence is folded in. The integration is two-phase — `preview`
computes updates without applying them, novelty is scored against that
pre-update bank, then `apply` commits — so a candidate is never rewarded for
novelty relative to a memory it just changed. The bank lives in
[`knowledge.py`](../src/esn/core/knowledge.py) /
[`knowledge_bank.py`](../src/esn/core/knowledge_bank.py).

---

## 4. From the memory to `N_sp`: the spectral mechanism

This is the heart of ESN. Once per generation it runs a random-matrix-theory
(RMT) analysis over the active hypotheses, and per candidate it computes a scalar
`N_sp ∈ [0,1]`. The intuition first, then the math.

### Intuition

Stack the hypotheses into a matrix, one row per hypothesis, each row its
embedding weighted by how *certain* the search is about that hypothesis. If the
search has genuinely learned structure, that matrix will have a few directions
that carry far more variance than random noise would — these are the **spikes**.
RMT gives us a principled way to say *which* eigenvalues are real signal and
which are just the noise you'd expect from a finite random matrix. The top
directions of the real signal span a "known structured subspace." A candidate's
`N_sp` is then simply: how much of *what it touched in memory* points *outside*
that subspace. (A candidate program has no embedding of its own — `N_sp` is
computed over the certainty-weighted average embedding of the hypotheses it
engaged.)

### The matrix and its spectrum

Build the **knowledge matrix** `K` with row `i = wᵢ·eᵢ`, where `eᵢ` is the
(PCA-compressed) embedding and the **certainty weight** is `wᵢ = 2·|cᵢ − 0.5|`.
Note what this weighting does: a *confirmed* hypothesis (`c≈0.9`) and a *refuted*
one (`c≈0.1`) both weigh strongly (`w≈0.8`), while an *untested* one (`c=0.5`)
weighs zero. The memory's structure is about *what the search is sure of*, in
either direction — not about belief polarity.

Center the matrix, then take its SVD, `K̃ = U·diag(σ)·Vᵀ`. The eigenvalues of the
sample covariance are `λⱼ = σⱼ² / H` (where `H` is the hypothesis count); `V`'s
columns are the candidate structural directions. Estimate the noise level as
`σ² = trace/d` and the aspect ratio `γ = d/H`. Marchenko–Pastur then predicts the
noise band edge `λ₊ = σ²(1 + √γ)²`, refined by a finite-sample Tracy–Widom
correction. Eigenvalues above the active threshold are spikes.

### Three threshold modes

How aggressively ESN calls something a spike is configurable
(`spectral_threshold_mode`):

- **`empirical` (default)** — a *shuffle null*. Permute the certainty↔embedding
  pairing 200 times, recompute the top eigenvalue each time, and take the 95th
  percentile as the threshold. This is conservative and data-driven; it makes no
  parametric assumption about the spectrum.
- **`mp`** — use the analytic Tracy–Widom-corrected Marchenko–Pastur edge
  directly.
- **`hybrid`** — use empirical, but fall back to the MP edge when the empirical
  threshold looks degenerate (more than 1.5× the analytic edge).

In the default `empirical` mode the MP/Tracy–Widom edge is computed but used only
as a diagnostic; the shuffle-null is what actually gates spikes.

### `N_sp` per candidate

Let `V_k` be the top-*k* structural directions (the known subspace). For a
candidate, take the certainty-weighted average embedding of the hypotheses it
engaged, center it against the stored mean to get `ẽ`, and project off the known
subspace:

```
N_sp = ‖ẽ − V_k V_kᵀ ẽ‖²  /  ‖ẽ‖²        (the residual energy fraction)
```

`N_sp = 1` means the candidate lies entirely outside known structure (fully
novel); `N_sp = 0` means it is fully explained by it.
([`compute_gram_schmidt_residual`](../src/esn/core/spectral.py).) Be aware of one
deliberately surprising default: when the subspace or mean is missing or the
centered vector is near zero, `N_sp` returns `1.0` ("treat as fully novel"), so on
a cold or empty memory everything reads as maximally new.

### What this means in practice (the gates)

`N_sp` is **off until structure is real and stable**. If the pipeline finds no
spikes, `N_sp` is suppressed to zero and the search runs on epistemic novelty
alone. Even once spikes appear, the spectral signal only starts mixing into
selection after spikes have *persisted for several consecutive generations* (a
signal-quality gate, default 3). The first generations of any run therefore
select almost purely on fitness and epistemic novelty; spectral steering switches
on only after the memory has accumulated durable structure. There is also a
richer **BBP calibration** layer (recovering each spike's true strength and an
eigenvector-alignment reliability). It leaves the legacy spike count and `V_k`
width unchanged, but its count of *actionable* spikes is unioned into the
**effective** spike count that gates the mixing weight — so it can switch
spectral novelty on earlier than the empirical detector alone would, and it also
feeds mutation-prompt guidance.

A practical consequence: full `N_sp` wants a real embedder, and there are two
distinct degraded paths worth knowing:

- **No analyzer at all** — the default. No hypotheses ever form, so there is no
  memory, both `N_sp` and `N_ep` stay zero, and selection reduces to **plain
  fitness**. This is the loud warning `esn.run` emits when you call it without an
  `analyzer`.
- **Analyzer present, but no `[novelty]` embedder** — hypotheses still form, so
  *epistemic* novelty still works, but embeddings collapse to zero vectors and
  the spectral signal disappears. Search runs on **fitness + epistemic novelty**.
  This is a separate warning.

---

## 5. Selection: the viability bar, and where novelty actually bites

Here is where it is worth being precise, because the headline framing and the
code diverge.

**The viability bar is real and strict.** Only `success=True` candidates can
become the run's best or enter an archive. A failing candidate is recorded for
analysis but is invisible to selection. (This is why your evaluator's
`success` flag matters — see
[connecting-a-problem.md → the evaluator contract](connecting-a-problem.md#the-evaluator-contract).)

**Best-promotion is pure fitness with a deadband.** The batch winner is simply
the highest-scoring successful candidate; it is promoted to the global best only
if it clears an improvement deadband of about 0.5% — `f_best + max(|f_best|,1)·0.005`
(note the floor of 1.0, which keeps the deadband meaningful even near zero).
On a real improvement, stagnation and temperature reset and a short
"breakthrough cooldown" biases the next few generations toward exploitation. On a
miss, stagnation rises and a search temperature ramps up. Novelty does **not**
tie-break or override fitness for best-promotion. The config fields
`selection_strategy="pareto"` and `fitness_epsilon` exist but are not consumed by
any code path — the "epsilon-band Pareto" name in the README captures the design
*intent* (spend the budget on the most novel option among the viable ones), and
the deadband is the closest literal analogue, but there is no code that picks the
most-novel candidate within a fitness band. Treat the README phrasing as
intuition, this section as the mechanism.

**Where novelty genuinely bites is everywhere *except* best-promotion.** This is
the substance of the "good *and* new" claim:

- **Archive routing.** Each successful candidate goes to one of two archives. If
  it scores within an elite band of the *running* best — `best_score·0.005`, and
  note this is the best as of the previous generation, since archiving happens
  before promotion — it joins the **elite archive** (a flat size-50 list,
  score-evicted). Otherwise it joins the **frontier archive** (a flat size-100
  list) *keyed on its unified novelty* — admission requires novelty (or
  repairability) ≥ 0.1, and eviction drops the least novel. The frontier is
  literally a novelty-ranked reservoir of viable-but-not-best ideas.
- **Parent selection.** The next generation's parents are drawn from the best,
  the top elites, **and** the most novel frontier members, then de-duplicated for
  diversity. So a candidate that wasn't the best but *was* novel gets to seed
  future mutations.

A concrete micro-example. Suppose the current best is `1.80` and this generation
produces three successes: A=`1.81` (a minor refinement, `N_sp≈0.0`), B=`1.79` (a
structurally different approach, `N_sp≈0.9`), C=`1.50` (`N_sp≈0.4`).

- A clears the deadband (`1.81 > 1.80 + 0.009`), so **A becomes the new best** —
  fitness wins, full stop.
- Archive routing is measured against the *pre-promotion* best (`1.80`), so the
  elite band is `1.80 − 0.009 = 1.791`. **A** (`1.81`) lands in the **elite
  archive**. **B** (`1.79`) falls just outside that band, so it routes to the
  **frontier**, admitted on its high novelty; **C** goes to the frontier too.
- When parents are picked next round, B's high frontier novelty makes it a prime
  exploration parent even though it scored *below* the old best — so the
  structurally new idea survives and propagates, precisely because the frontier
  is novelty-keyed, while a naive loop would have discarded it.

That is the mechanism by which ESN avoids collapse: fitness decides the
*champion*, but novelty decides what *lives on to be explored*.

---

## 6. The rest of the machinery that shapes search

Several smaller systems shape where the search spends effort. Each relates back
to the core "good and new" idea.

**Epistemic novelty `N_ep`.** The other half of the blended novelty. Where `N_sp`
is geometric, `N_ep` is *belief-revision*: it rewards candidates whose evidence
moved confident hypotheses (`Σ cᵢ·δᵢ`) plus a small bonus for introducing new
hypotheses and for **prediction surprise** — a bit set when the score lands
outside the predictor's pre-evaluation estimate. Broken candidates have their
`N_ep` heavily discounted. `N_ep` is min-max normalized to `[0,1]`.

**The unified blend.** Selection-facing novelty is
`N = γ·N_sp + (1−γ)·N_ep`, with `γ = sigmoid(−erank/τ)` — *lower effective rank*
(more concentrated structure) leans toward spectral, more diffuse memory leans
toward epistemic. As noted in §4, `γ` is hard-gated to zero until spikes persist,
so early on `N` is purely epistemic. ([`scorer.py`](../src/esn/core/scorer.py).)

**Operator credit (which mutation *style* to try).** The bandit arms are the four
core styles — refine, explore, repair, radical. A UCB rule with an ε-greedy floor
picks among them, rewarding by recent score improvement in EXPLOIT mode and by
*epistemic novelty* in EXPLORE mode. (Two further multi-parent styles sit outside the bandit:
`synthesize` is reserved for the BRIDGE mode — a latent mode the current selector
does not emit — and `recombine` is allocated only by the gated recombination path
below; neither is UCB-sampled.)
Credit is attributed using the **raw** LLM score (before any deterministic local
polish), so the bandit learns about the *mutator's* ideas, not the
post-processing.

**Search modes.** A small priority cascade maps the run's state to a mode —
heavy recent failure → REPAIR; a clean winning streak → EXPLOIT; deep stagnation
with low diversity → EXPLORE; and so on — with engine overrides (a breakthrough
cools toward EXPLOIT; deep stagnation forces EXPLORE). The mode selects which
mutation styles are even on the table for the next batch.

**Branches and families.** Every candidate's lineage is tracked as a **branch**;
branches are bucketed into coarse structural **families** (recursive-multi,
iterative-flat, …) via a *deterministic AST fingerprint* — not a neural
embedding. The key diversity rule: a child forks a new branch *unconditionally*
when its family changes (an emerging strategy is never absorbed into the dominant
branch), and otherwise only when it is both good enough and geometrically far
from its branch centroid. Branches retire on stagnation or domination, capped at
a small live set, so the search maintains a portfolio of distinct lines of attack.

**Recombination.** An opt-in operator (`enable_recombination`, off by default):
on a global plateau, it asks the LLM to fuse two diverse high-performing branches
into one solver. It fires only behind several gates — enough live branches, a
sustained plateau, two genuinely diverse high-quality parents, and a cooldown —
so it is a targeted "combine the best of two ideas" move, not a constant cost.

---

## 7. A worked example: `circle_packing`

To make the mechanism concrete, here are real artifacts from one short run of the
bundled [`circle_packing`](../examples/circle_packing) domain (pack 26
non-overlapping circles in a unit square; score = sum of radii) — **6
generations, batch size 2, seed 42**, driven key-free by Claude Haiku. The seed
is a 1‑8‑17 concentric-ring layout scoring **1.6602**; after six generations the
best reaches **1.7435**.

> LLM steps are nondeterministic, so a re-run won't reproduce these exact
> hypotheses or scores. This capture also ran *without* the `[novelty]` embedder,
> so `N_sp` reads `0.0` throughout and novelty here is **epistemic-only** (§4) —
> install `--extra novelty` for the live spectral signal. To run your own:
>
> ```bash
> uv run python examples/run.py --domain circle_packing \
>     --mutator agent --analyzer agent --generations 6 --batch-size 2 --seed 42
> ```

### The knowledge bank after the run

The analyzer distilled the six generations into **15 active hypotheses**. Each is
a short causal claim with a Beta-Bernoulli `confidence` and an observation count
`n_obs` (§3). A representative slice, highest-confidence first:

| `confidence` | `n_obs` | hypothesis (abridged) |
|:---:|:---:|---|
| 0.75 | 2 | Anisotropic ring compression — exploit the square's corners by compressing outer-ring circles toward them (4-fold pattern), reducing required radii. |
| 0.63 | 4 | Alternative ring partitions (1‑6‑19 or 1‑12‑13) with recalibrated radii beat the 1‑8‑17 topology. |
| 0.50 | 3 | Concentric-ring topology is a local-maximum trap; multi-start / simulated annealing on positions finds denser packings. |
| 0.50 | 1 | Boundary-contact maximization — pre-place circles on edges and corners (4-fold symmetry), then pack the interior. |
| 0.38 | 4 | Non-uniform outer ring (corner clustering / edge-hugging) recovers density lost to symmetric placement. |
| 0.25 | 2 | Ring–spiral hybrid: outer rings in an Archimedean spiral, the inner circle optimized independently. |
| 0.17 | 3 | Number-theoretic packing exploiting the factors of 26 (2×13, 1+25) with a distinct strategy per group. |

**This table *is* the memory `N_sp` is measured against** (§3–§4): each row becomes
a certainty-weighted row of the knowledge matrix. The confidence spread is the
Beta-Bernoulli update at work — the corner-compression idea was supported by two
candidates (→ 0.75), while the number-theoretic-factoring idea was mostly refuted
across three (→ 0.17). Each stored hypothesis also carries an appended
`[aspects: family=… | operator=… | motifs=… | score=… ]` tag (added before
embedding) so structurally-distinct ideas stay distinguishable. Separately, over
this run the `explore` style earned the highest mean epistemic novelty (0.67) of
the four bandit arms — exactly the signal it is rewarded on in EXPLORE mode (§6).

### A mutation, end to end

The most vivid mutation was an **`explore`** step in generation 1. Its parent was
the **seed** (the concentric-ring construction, score **1.66**); the child
abandoned rings entirely for a **golden-angle (Fibonacci) spiral**:

Parent — seed (excerpt):

```python
# Inner ring: 8 circles at radius 0.25; outer ring: 17 at radius 0.42
for i in range(8):
    angle = 2 * np.pi * i / 8
    centers[1 + i] = [0.5 + 0.25*np.cos(angle), 0.5 + 0.25*np.sin(angle)]
for i in range(17):
    angle = 2 * np.pi * i / 17 + np.pi / 17
    centers[9 + i] = [0.5 + 0.42*np.cos(angle), 0.5 + 0.42*np.sin(angle)]
```

Child — `explore` (excerpt):

```python
golden_angle = 2.39996322972865          # ≈ 137.5°, the golden angle
for i in range(n):
    theta = i * golden_angle
    r = 0.42 * np.sqrt(i / (n - 1.0) + 0.05)   # spiral radius grows ∝ √i
    centers[i] = [0.5 + r*np.cos(theta), 0.5 + r*np.sin(theta)]
# … then iterative overlap resolution + a radius-growth phase
```

The engine recorded this candidate as `style=explore`, `family=iterative-nested`,
`score=1.3407`, `epistemic_novelty=1.0` (it pushed genuinely new structure into
the memory). Note its score — **1.34, *below* the seed's 1.66**. A greedy
"keep-the-best" loop would discard it on the spot. ESN does not: as a
high-novelty success it is admitted to the **novelty-keyed frontier** (§5), and
in generation 2 it became the parent of a further `explore` that climbed to
**1.4532** — a whole second lineage that only exists because the below-seed idea
was kept alive.

Meanwhile fitness crowned a different lineage entirely: a quieter **`refine`**
step (same generation, same seed parent) kept the ring structure and only
retuned it — inner radius `0.25 → 0.232`, outer `0.42 → 0.413`, and the
overlap-resolution loop `5 → 25` iterations with a tighter tolerance. That
refine→repair→refine chain is what reached the run's best of **1.7435**.

That split is the whole mechanism in one run: **fitness decided the champion (the
refinement chain), novelty decided what else stayed alive to be explored (the
spiral) — and a naive loop would have kept only the first.**

---

## 8. Why it's efficient

The scarce resource in this kind of search is the **LLM call** — each mutation and
analysis costs latency and money, while compiling and scoring a candidate in a
`uv` subprocess is cheap by comparison. "Efficient" therefore means reaching a
good solution in **fewer expensive creative steps**, and every part of the design
spends *cheap* computation to make each *expensive* call count.

- **It stops paying for the same idea twice.** A greedy loop collapses onto one
  approach and burns calls re-deriving variations of it. Scoring structural
  novelty against the memory — and deduping that memory — routes budget away from
  near-duplicates the model has already produced.
- **No viable call's output is wasted.** Every success, even one *below* the
  current best, is kept on the novelty-ranked frontier and can become a parent.
  In §7 the golden-spiral scored 1.34 — below the 1.66 seed; a greedy loop deletes
  it and loses the call that made it, whereas ESN turned it into a second lineage
  that reached 1.45. "Didn't improve" becomes exploration fuel instead of waste.
- **Each call is better-informed.** Hypotheses distilled from past candidates (§3)
  feed the next mutation prompt, so the model proposes *conditioned on what the
  search has already learned* rather than starting cold — raising the hit-rate per
  call, at no extra LLM cost (the belief updates are arithmetic, §6).
- **Cheap compute aims the expensive budget.** The spectral analysis, the UCB
  operator-credit bandit, mode selection, and the adaptive batch controller all
  run *between* calls. They steer the next call (which parent, which style, explore
  vs. exploit), stop funding mutation styles that aren't working, conserve batch
  budget by default, and snap back to a known-good program after repeated failures
  rather than flailing.
- **It escapes local optima without crashing fitness.** The viability bar keeps it
  out of infeasible regions while novelty keeps it off a single peak — the balance
  that reaches a better optimum in fewer generations than pure exploitation (which
  gets stuck) or pure exploration (which never converges).

**The trade-off, honestly:** the analyzer (and optional predictor) add LLM calls
*per candidate*, so ESN spends somewhat more per step than a bare loop. The bet is
that the extra bookkeeping buys far better targeting — fewer steps to escape traps
and reach a good answer — and net-saves the expensive budget. It pays off most
when the search space is wide and greedy loops collapse; on easy or narrow
problems the overhead may not earn its keep.

---

## 9. The sandbox and persistence

**Why candidates run isolated.** Every candidate is *untrusted LLM-generated
code*. The default `UvSandboxCompiler` runs each candidate in its own `uv run`
subprocess: the code is piped to a runner, executed, `solve()` is called, and the
result is emitted as JSON after a sentinel line (with markers that survive the
tuple→list and numpy round-trips). A process-wide semaphore caps concurrent
subprocesses to the CPU count to prevent thrashing. There is also an in-process
`PythonSandboxCompiler` (restricted builtins, no imports, `SIGALRM` timeout —
best-effort, Unix-only) and a `StdioCompiler` for stdin→stdout programs where the
stdout text *is* the artifact. All three pass an AST gate first that rejects
sandbox-escape patterns (`exec`/`eval`/`__import__` and dunder-attribute
escapes). The in-process `PythonSandboxCompiler` additionally enforces an import
allow-list and rejects async via the gate; the default `UvSandboxCompiler` skips
import checks at the AST level (it relies on subprocess isolation and declared
dependencies instead), and the stdio gate is lighter still. (Which to choose:
[connecting-a-problem.md](connecting-a-problem.md#choose-solve-vs-stdio-and-the-compiler).)

The same AST pass also produces the **structural fingerprint** (family, feature
tags, a control-flow hash) that becomes the 38-dimensional vector branch geometry
uses — a deterministic, domain-free "embedder" with no neural model in the loop.

**What resumes.** A run checkpoints to a directory of plain JSON files. Each
concern has its own store and each loads independently, so a partial checkpoint
recovers what it can: the search state (generation, best, stagnation), both
archives, the operator-credit bandit memory, the hypothesis bank (embeddings
base64-encoded), and the spectral state with its spike history. Non-serializable
pieces — the embedder, the novelty observation hooks — are re-attached from the
live objects on load. The practical upshot: you can stop and resume a search and
the bandit, the memory, and the spectral structure all come back warm.

---

## 10. The protocol seams: how the engine stays domain-agnostic

Everything above is held together by a handful of small, swappable seams. The
formal `Protocol`s — `Mutator`, `ProgramCompiler`, `Predictor`, `Analyzer` (and
`LocalImprover`) — define interfaces the engine depends on without binding to any
concrete implementation, which is what lets you swap any one part without
touching the rest:

- **`Mutator`** — `mutate(parents, style, context) → MutationResult`. Proposes
  candidates. Ships as a single-shot LLM mutator and an agentic Claude-SDK
  mutator. See [mutators.md](mutators.md).
- **`ProgramCompiler`** — `compile(code, seed) → CompilerResult`. Turns code into
  a runnable artifact under sandboxed, bounded execution. Three bundled
  implementations (§9).
- **`Analyzer`** — turns an evaluated candidate into evidence + new hypotheses,
  feeding the memory that `N_sp` is measured against (§3).

The **`NoveltyComputer`** is, strictly, a concrete class rather than a `Protocol`
— the engine consumes it duck-typed (`Any`), so it is still swappable in
practice. It owns the spectral pipeline and returns `(epistemic, spectral,
unified)` per candidate; with no analyzer/memory it cleanly degrades to zeros and
the search runs on fitness alone.

Plus `Predictor`, an `OperatorCreditModel`, `LocalImprover`, and a batch slot
scorer, all injectable. Everything domain-specific lives in **one object**, the
`DomainSpec` — seed program, compiler, evaluator, and prompt-steering hints. The
engine itself never changes.

To put ESN on your own problem, write that one `DomainSpec`:
**[connecting-a-problem.md](connecting-a-problem.md)**. To choose how mutations
are proposed: **[mutators.md](mutators.md)**. For the high-level pitch and
install paths: **[README](../README.md)**.
