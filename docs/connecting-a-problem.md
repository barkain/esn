# Connect your own problem to ESN

You apply `esn` to a new problem by writing **one object: a `DomainSpec`**. The
engine is fully domain-agnostic — it never changes. The `DomainSpec` is the
complete contract between your problem and the search.

A `DomainSpec` bundles five things: a **description** of the problem, a **seed
program** (`initial_code`), a **compiler** that runs candidate code in a sandbox,
an **evaluator** that scores the result (higher = better), and a few **NL
hints/constraints** that steer the mutator.

## Choose `solve()` vs `stdio` (and the compiler)

This decides `program_interface` and which compiler you pass:

| Your answer is… | `program_interface` | Compiler | Example |
|-----------------|---------------------|----------|---------|
| an in-process Python value (array, list, dict) | `"solve"` (default) | `UvSandboxCompiler` (default; isolated `uv run`) or `PythonSandboxCompiler` (in-process) | [`circle_packing`](../examples/circle_packing) |
| stdin/stdout text (an I/O protocol, e.g. AHC/ALE-Bench) | `"stdio"` | `StdioCompiler` | [`tsp`](../examples/tsp) |

All three compilers are top-level `esn.*` exports satisfying the
`ProgramCompiler` protocol (`compile(code, seed) -> CompilerResult`).
**Import allowlists and line limits live on the compiler**, not the `DomainSpec`
(e.g. `UvSandboxCompiler(allowed_imports=frozenset({"numpy"}), max_lines=...)`).

## Fields

**Required:**

| Field | Kind | What it is |
|-------|------|------------|
| `name` | mechanical | Short identifier for the domain. |
| `description` | prompt-facing | One-paragraph statement of the problem. |
| `initial_code` | mechanical | The seed program the search mutates from. |
| `compiler` | mechanical | A `ProgramCompiler` that runs candidate code (see above). |
| `evaluator` | mechanical | `(artifact) -> EvaluationResult`; returns the score. `EvaluationResult` includes optional diagnostic metadata — for simple scorers you may ignore the `diagnostics` field. |

**Key optional, prompt-facing:**

| Field | Default | What it is |
|-------|---------|------------|
| `program_interface` | `"solve"` | `"solve"` or `"stdio"` (see above). |
| `hard_constraints` | `[]` | Rules the solution must obey; rendered into the prompt. |
| `hints` | `[]` | Advisory nudges for the mutator. |
| `examples` | `[]` | Worked examples shown to the mutator. |
| `preferred_solution_shape` | `None` | Advisory steer toward the *shape* of a good solution (e.g. "prefer constructive solutions over long search"). No compile/eval effect. |

(`style_overrides` — per-style prompt overrides. Import allowlists and line
limits are set on the **compiler**, not here — see above.)

## The evaluator contract

Your evaluator is `Callable[[artifact], EvaluationResult]`. It receives the
**compiled artifact** — the value `solve()` returned, or the stdout text for
`stdio` — and returns `EvaluationResult(score=<float>, success=<bool>, diagnostics=<optional>)`.

**`score` MUST be higher-is-better.** If you minimize (tour length, error, cost),
**invert it** — return `1.0 / x` or `-x` — so larger = better:

```python
return esn.EvaluationResult(score=1.0 / tour_length, success=True)  # minimizing length
```

Forgetting to invert does not error: the search **silently optimizes the worst**
solution. Set `success=False` (and usually `score=0.0`) for infeasible or crashed
candidates — **a `success=False` candidate can never become the best** (the engine
only tracks the best among successful ones).

## Minimal end-to-end example

```python
import esn


def evaluate(artifact) -> esn.EvaluationResult:
    # `artifact` is whatever solve() returned. Higher score = better.
    if not isinstance(artifact, (int, float)):
        return esn.EvaluationResult(score=0.0, success=False)
    return esn.EvaluationResult(score=float(artifact), success=True)


domain = esn.DomainSpec(
    name="my_task",
    description="Return the largest value f(x) you can under the constraints.",
    initial_code="def solve():\n    return 0.0\n",
    compiler=esn.UvSandboxCompiler(allowed_imports=frozenset({"math"})),
    evaluator=evaluate,
    program_interface="solve",
    hard_constraints=["solve() must return a single float."],
    hints=["Construct a feasible answer first, then improve it."],
)

# Run it with a real mutator + analyzer (key-free, Claude subscription):
result = esn.run(
    domain,
    mutator=esn.make_agent_mutator(domain),
    analyzer=esn.make_agent_analyzer(),
)
print(result.best_score, result.best_code)
```

For full worked specs, see [`examples/circle_packing`](../examples/circle_packing)
(a `solve()` domain) and [`examples/tsp`](../examples/tsp) (a `stdio` domain).

## Enabling novelty (the `N_sp` signal)

ESN steers search with a spectral-novelty signal, `N_sp`. Passing an **analyzer**
to `esn.run(...)` activates the novelty machinery (hypotheses + epistemic novelty);
without one, `esn.run` warns loudly that novelty is inactive. The spectral `N_sp`
signal additionally needs the `[novelty]` embedder — without it embeddings are
zero and `N_sp` stays flat (epistemic novelty still works). Pick one line:

```python
result = esn.run(domain, analyzer=esn.make_agent_analyzer())                # key-free
result = esn.run(domain, analyzer=esn.make_analyzer(model="gpt-4o"))        # keyed
```

`make_agent_analyzer()` is key-free (`uv sync --extra agent`, Claude subscription);
`make_analyzer(model=...)` uses an API key (`--extra llm`). Add `--extra novelty`
for the full embedding-based `N_sp`. (Full table: [README → Credentials](../README.md#credentials--api-keys).)

For how `N_sp` is actually computed from the hypothesis memory and how it steers
selection, see **[how-it-works.md](how-it-works.md)**.

## Passing instance data

`solve()` takes no arguments, so problem data is **baked into two places that must
stay in sync**: module-level constants in `initial_code` (the mutator sees and uses
them) and the same values closed over by your `evaluator`. circle_packing inlines
its single instance (`N_CIRCLES = 26` in both); for many instances or train/val
splits, load them from a bundle the evaluator closes over —
see [`examples/tsp/instance_bundle.py`](../examples/tsp/instance_bundle.py).
