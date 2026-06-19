# Choosing a mutator: single-shot LLM vs agentic

The mutator is what *proposes* each new candidate. ESN supports two real
mutators, and they are **fully interchangeable** — both implement the same
`Mutator` protocol and are passed the same way:

```python
result = esn.run(domain, mutator=mutator, generations=20)
```

The engine is mutator-agnostic, so you can swap one for the other without
touching your `DomainSpec` or the engine.

## Comparison

| | **Single-shot LLM** | **Agentic (Claude Agent SDK)** |
|---|---|---|
| How it works | one LLM completion per mutation | a multi-turn Claude *agent* session per mutation |
| Latency | fast | slower (multi-turn) |
| Cost | cheap, predictable | heavier |
| Capability | strong for incremental edits | richer; can reason/iterate, optionally use research tools |
| Extra | `[llm]` | `[agent]` |
| Auth | API key in env: `gpt-*` → `OPENAI_API_KEY`, `claude-*` → `ANTHROPIC_API_KEY` | Claude subscription / keychain (no env var needed), or optionally `ANTHROPIC_API_KEY` |
| Best for | fast/cheap iteration (the right default) | harder problems, research-augmented mutation |

The single-shot LLM path is the **original design and the right default** for
most runs: fast, cheap, and predictable. Reach for the agent path when a problem
needs deeper per-step reasoning or web research.

## Single-shot LLM

One LLM call per mutation. Provider is inferred from the model-name prefix
(`gpt-*`/`o*` → OpenAI, `claude-*` → Anthropic). Needs the `[llm]` extra and the
matching API key in your environment.

```python
import esn

mutator = esn.make_llm_mutator(domain, model="gpt-4o")
result = esn.run(domain, mutator=mutator, generations=20)
```

Works with the bundled `circle_packing` example (e.g. `gpt-4o-mini`).

### Edit format: full-rewrite vs diff

The single-shot mutator has two **edit formats**, selected by `mutator_policy`:

- `"single_shot"` (default) — the LLM regenerates the **whole program** each
  mutation. Good at large structural jumps (e.g. swapping a constructor for a
  numerical optimizer in one step).
- `"diff"` — the LLM emits Aider-style `SEARCH`/`REPLACE` blocks that are applied
  to the parent, so the rest of the program is preserved verbatim. Good at
  **incremental** edits that accumulate down a lineage.

```python
mutator = esn.make_llm_mutator(domain, model="gpt-4o", mutator_policy="diff")
```

Which wins is task-dependent: diff shines when progress is a sequence of small
local refinements; full-rewrite shines when the breakthrough is a wholesale
restructuring (on `circle_packing` with a weak model, full-rewrite reached higher
because the key move is a large constructor→optimizer rewrite). Diff edits are
also more fragile on weak models — a `SEARCH` that no longer matches, or an edit
that breaks validity, is dropped. The mutator records `diff_changed_frac` and
`diff_full_rewrite` in its metadata so you can confirm edits are genuinely
incremental rather than silently regenerating.

`--max-tokens` (CLI) / `max_tokens=` (`make_llm_mutator`) raises the completion
cap from the 1024 default — necessary for domains whose programs are long, or
they get truncated.

## Agentic (Claude Agent SDK)

A multi-turn Claude agent session per mutation. Optionally consult web tools
with `mutator_tools="research"` (WebSearch / WebFetch, behind an isolation
boundary). Needs the `[agent]` extra; authenticates via your Claude subscription
/ keychain (no env var needed), or optionally `ANTHROPIC_API_KEY`.

`model` is optional and defaults to `"claude-haiku-4-5-20251001"`:

```python
import esn

# explicit model:
mutator = esn.make_agent_mutator(domain, model="claude-haiku-4-5-20251001")
# or omit model to use the default:
# mutator = esn.make_agent_mutator(domain)
# research-augmented variant:
# mutator = esn.make_agent_mutator(domain, mutator_tools="research")
result = esn.run(domain, mutator=mutator, generations=20)
```

Works with the bundled `circle_packing` and `tsp` examples.

## Driving novelty: the analyzer

Passing an `analyzer=` to `esn.run` activates the novelty machinery (hypotheses +
epistemic novelty); without it, `esn.run` warns loudly that novelty is inactive.
The spectral `N_sp` signal additionally needs the `[novelty]` embedder (below).
The analyzer comes in the same two tiers as the mutator:

```python
analyzer = esn.make_agent_analyzer()            # key-free, Claude subscription, [agent]
# analyzer = esn.make_analyzer(model="gpt-4o")  # keyed, [llm]
result = esn.run(domain, mutator=mutator, analyzer=analyzer, generations=20)
```

Add the `[novelty]` extra (sentence-transformers, no key) for the learned
embeddings that give a real `N_sp`; without it, embeddings are zero vectors and
`N_sp` stays flat (epistemic novelty still works). Credentials per
component: [README → Credentials](../README.md#credentials--api-keys).

When you pass an analyzer, the bundled `examples/run.py` also wires a matching
**predictor** by default (the prediction-surprise term of epistemic novelty);
`--no-predictor` turns it off. An optional `--tune`
([`ParameterTuner`](../src/esn/engine/tuner.py)) adds evaluator-guided
continuous-parameter polish — useful when solution quality is driven by float
constants, a safe no-op otherwise.

See [connecting-a-problem.md](connecting-a-problem.md) for how to build the
`DomainSpec` these mutators operate on, and
[how-it-works.md](how-it-works.md) for how the analyzer's hypotheses feed the
spectral-novelty signal that biases selection.
