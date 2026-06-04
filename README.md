# ESN - Epistemic-Spectral-Novelty

[![CI](https://github.com/barkain/esn/actions/workflows/ci.yml/badge.svg)](https://github.com/barkain/esn/actions/workflows/ci.yml)

**Spend your LLM budget on candidates that are both *good* and *genuinely new*.**

---

## What it is

`esn` runs **LLM-driven evolutionary search**: an LLM proposes mutations
of a candidate solution, those candidates are compiled and scored, and the best
ones become parents for the next round. That much is the familiar "LLM in a
loop" pattern.

The difference is *what gets selected*. A naive loop keeps whatever scored
highest, so it quickly collapses onto one idea and burns expensive LLM calls
re-deriving variations of the same approach. `esn` instead steers
selection with an **epistemic spectral-novelty signal**: it maintains a memory
of the structures it has already learned, runs a spectral analysis over that
memory, and measures how *structurally unlike* each new candidate is from
everything understood so far (the spectral-novelty score, `N_sp`).

Selection then uses an **epsilon-band Pareto rule**: among candidates whose
fitness is within a small band of the current best (`f >= f_max - epsilon`),
the *most novel* one wins. Candidates have to clear the viability bar to be
considered at all, but within that bar the search deliberately spends its next
LLM call on the option that teaches it something new. The result is exploration
that does not crash fitness and exploitation that does not stagnate.

---

## Install

This project is **uv-only** — [`uv`](https://docs.astral.sh/uv/) is both the
installer *and* the sandbox runtime that executes every candidate the search
generates, so it must be on your `PATH` (check with `uv --version`). Install
from source:

```bash
git clone https://github.com/barkain/esn.git
cd esn
uv sync   # offline core (pydantic + numpy) — no LLM deps
```

Bare `uv sync` installs the **offline core**. Add the extra each path needs:
`--extra agent` (key-free Claude subscription), `--extra llm` (OpenAI/Anthropic
API keys), `--extra novelty` (learned-embedding `N_sp`). Requires Python ≥ 3.10.

---

## Quickstart

### Agentic Mutation

A multi-turn Claude *agent* proposes each mutation. `make_agent_mutator` +
`make_agent_analyzer` authenticate through your local Claude install / macOS
keychain — **no API key**. Pass a real `analyzer` so novelty (`N_sp`) is active
(without one `esn.run` warns and degrades to plain fitness search).

```bash
uv sync --extra agent --extra novelty   # subscription SDK + learned-embedding N_sp
```

```python
import esn
from your_domain import MY_DOMAIN  # a DomainSpec you define (see below)

mutator  = esn.make_agent_mutator(MY_DOMAIN, model="claude-haiku-4-5-20251001")
analyzer = esn.make_agent_analyzer(model="claude-haiku-4-5-20251001")  # hypotheses → N_sp novelty
result = esn.run(MY_DOMAIN, mutator=mutator, analyzer=analyzer, generations=20, seed=42)

print(result.best_score, result.best_code)
```

### Linear Prompt-Response Mutation

One LLM completion proposes each mutation — faster and cheaper, billed to a
provider API key chosen by model name (`gpt-*` → `OPENAI_API_KEY`, `claude-*` →
`ANTHROPIC_API_KEY`):

```bash
uv sync --extra llm --extra novelty
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY=... for a claude-* model
```

```python
mutator  = esn.make_llm_mutator(MY_DOMAIN, model="gpt-4o")
analyzer = esn.make_analyzer(model="gpt-4o-mini")
result = esn.run(MY_DOMAIN, mutator=mutator, analyzer=analyzer, generations=20, seed=42)
```

See [docs/mutators.md](docs/mutators.md) for when to use which.

## Credentials / API keys

| Component | Factory | Key | Extra |
|---|---|---|---|
| Agentic mutator/analyzer (subscription) | `make_agent_mutator` · `make_agent_analyzer` | **none** (local Claude / keychain) | `agent` |
| LLM mutator/analyzer/predictor | `make_llm_mutator` · `make_analyzer` · `make_predictor` | `gpt-*`/`o*` → `OPENAI_API_KEY`, `claude-*` → `ANTHROPIC_API_KEY` | `llm` |
| Embedder (novelty) | auto (once an analyzer is passed) | none (local model) | `novelty` |

## Core concepts

- **`N_sp` (spectral-novelty score)** — how *structurally unlike* a candidate is from everything learned so far; the signal that steers selection.
- **Hypothesis** — what the analyzer extracts from each evaluated candidate; the memory `N_sp` is measured against.
- **Spectral analysis** — the decomposition over that hypothesis memory used to compute `N_sp`.
- **Epsilon-band Pareto** — among candidates within a small fitness band of the best (`f ≥ f_max − ε`), pick the *most novel* one.

---

## Use it on your own problem

You apply `esn` to a new problem by writing **one object: a `DomainSpec`** — a
problem description, a seed program, a sandbox compiler, an evaluator that scores
candidates (higher = better), and a few natural-language hints. The engine is
domain-agnostic and never changes.

See **[docs/connecting-a-problem.md](docs/connecting-a-problem.md)** for the
fields, the `solve` vs `stdio` interfaces, the evaluator contract, and a minimal
copy-paste example.

## Mutators

The mutator proposes each candidate. ESN supports a **single-shot LLM** mutator
(`esn.make_llm_mutator`, the fast/cheap default) and an **agentic** Claude Agent
SDK mutator (`esn.make_agent_mutator`, for harder / research-augmented runs).
They are interchangeable via `esn.run(domain, mutator=...)`.

See **[docs/mutators.md](docs/mutators.md)** for the comparison and one-liners.

---

## Bundled examples

Two complete `DomainSpec` implementations ship in [`examples/`](examples/) as
working references:

- [`examples/circle_packing/`](examples/circle_packing) — pack *n* circles into a
  unit square to maximize the sum of radii. A continuous-geometry domain;
  good for seeing exploration vs. exploitation trade-offs.
- [`examples/tsp/`](examples/tsp) — travelling-salesman tour minimization over the
  bundled instances. A combinatorial domain with a `stdio` program interface.

Each example is a self-contained template: copy the directory, swap in your
problem's `description`, `initial_code`, `evaluator`, and constraints, and you
have a new domain.

---

## Architecture

The engine is **domain-agnostic** and composed of pluggable parts. You provide a
`DomainSpec`; everything else is swappable behind a small set of protocols.

```
DomainSpec ─┐
            │   ┌──────────────────────────────────────────────┐
 Mutator ───┼──▶│  ESNEngine                                  │
            │   │   1. mutate parents      (Mutator)            │
 Compiler ──┤   │   2. compile candidate   (ProgramCompiler)    │
            │   │   3. evaluate → fitness  (DomainSpec.evaluator)│
 Novelty ───┘   │   4. score novelty N_sp  (NoveltyComputer)    │
                │   5. epsilon-band Pareto select               │
                │   6. update memory / archives → loop          │
                └──────────────────────────────────────────────┘
```

Pluggable seams (Python `Protocol`s):

- **Mutator** — proposes new candidates. `make_agent_mutator` (key-free
  agentic) or `make_llm_mutator` (single-shot LLM).
- **Compiler** (`ProgramCompiler`) — turns candidate code into a runnable
  artifact. The bundled uv-subprocess compiler isolates each candidate in its
  own `uv run` environment.
- **Novelty** (`NoveltyComputer`) — the spectral-novelty signal that scores how
  structurally new a candidate is relative to learned memory.

Swap any one of these without touching the others; the engine only depends on
the protocols.

---

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE).

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{esn,
  title  = {ESN: Epistemic Spectral Novelty},
  author = {Barkai, Nadav},
  year   = {2026},
  url    = {https://github.com/barkain/esn}
}
```
