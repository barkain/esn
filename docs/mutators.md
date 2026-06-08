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

Proven on `circle_packing` with `gpt-4o-mini`.

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

Proven on `circle_packing` (1.6602 → 1.8003) and `tsp`.

## Driving novelty: the analyzer

Spectral novelty (`N_sp`) only activates when you pass an `analyzer=` to
`esn.run` — without it, `esn.run` warns loudly that novelty is inactive. The
analyzer comes in the same two tiers as the mutator:

```python
analyzer = esn.make_agent_analyzer()            # key-free, Claude subscription, [agent]
# analyzer = esn.make_analyzer(model="gpt-4o")  # keyed, [llm]
result = esn.run(domain, mutator=mutator, analyzer=analyzer, generations=20)
```

Passing an analyzer also auto-activates the local embedder; add the
`[novelty]` extra (sentence-transformers, no key) for full `N_sp`. Credentials per
component: [README → Credentials](../README.md#credentials--api-keys).

See [connecting-a-problem.md](connecting-a-problem.md) for how to build the
`DomainSpec` these mutators operate on, and
[how-it-works.md](how-it-works.md) for how the analyzer's hypotheses drive the
spectral-novelty signal that steers selection.
