# Local SQLi lab — applying ESN to authorized security testing

This example applies ESN to **authorized security testing**: it evolves a SQL
injection (CWE-89) that extracts a hidden secret from a deliberately vulnerable
endpoint. Everything is **self-contained and offline** — there is no external
target, and the example is safe to run in CI.

## Safety

- The target is an **in-process, in-memory SQLite** app that lives only inside
  the evaluator. It is intentionally vulnerable, but it cannot reach the network,
  the filesystem, or any external system.
- Candidate `solve()` is asked to be **pure** — build a plan of probe *attempts*,
  do no I/O — and the evaluator performs every request, enforces scope, bounds SQL
  cost/allocation, and rejects destructive (`DROP`/`DELETE`/…) and nondeterministic
  (`random()`/…) SQL.
- **Isolation boundary (what is guaranteed):** the candidate runs in an isolated
  `uv` **subprocess** (`UvSandboxCompiler`) and returns its artifact as JSON, so it
  **cannot read the evaluator's per-process secret or otherwise game the score** —
  that is the security-relevant guarantee, and it is what makes the scoring
  non-gameable. Like ESN's other `UvSandboxCompiler` examples, the candidate is
  *not* additionally restricted from filesystem/network I/O **within** that
  subprocess (the "no I/O" rule is a prompt instruction, not an OS sandbox). If you
  run fully-untrusted candidates, wrap execution in an OS-level sandbox.
- For ESN research and CI only. To point a *separate, advanced* variant at a real
  target you are authorized to test, you would keep `solve()` pure and add a
  network adapter behind an explicit authorization gate in the evaluator — that
  is deliberately **not** part of this bundled example.

## How it works

The candidate returns `{"target": "local_sqli_lab", "attempts": [...]}`. Each
attempt is one of:

| shape | what it is |
|---|---|
| `{"payload": "..."}` | a direct SQL payload (syntax / error / timing probes) |
| `{"boolean_template": "... {predicate} ..."}` | the evaluator substitutes a true/false predicate and credits only a **measured** row-count flip |
| `{"payload_template": "... {n} ... {qprefix} ...", "charset": "...", "max_depth": N}` | a **blind-extraction** template the evaluator drives itself |

The evaluator scores a continuous **distance-to-exploit** ladder (0–1000):
valid plan → reaches the SQL parser → real SQLite error → boolean differential →
timing primitive (a case-folded `delay_ms` past the toy filter) → blind canary
recovery (+25 per genuinely-recovered character) → full secret recovered.

**The scorer is non-gameable.** Every rung is derived from *measured target
behavior*, never from candidate metadata. Full credit (`1000`) is awarded only
when the **evaluator itself** reconstructs the entire secret character-by-character
through the candidate's blind template, verified per character against a
**primary / wrong-control / shadow-target** differential, with the secret a
per-process random value the candidate never sees. So echoing, hard-coding,
`CASE WHEN`-ing, tautologies, `random()` oracles, and metadata guesses all earn
only the base score — the only way up the ladder is a genuine, secret-dependent
injection.

## Run it

```bash
# offline smoke (no API key, MockMutator):
uv run pytest tests/test_local_sqli_lab.py -q

# evolve a real exploit (key-free Claude subscription; needs the [agent] extra):
uv run python examples/run.py --domain local_sqli_lab \
    --mutator agent --analyzer agent --generations 30 --batch-size 4 --seed 42
```

## A note on novelty

ESN's branch/family machinery preserves diversity in *payload-generator program
architecture* (loops vs recursion, ranking, accumulators) — it does **not**
classify SQLi techniques. Technique diversity (boolean vs timing vs union) is
driven by the LLM mutator and by the analyzer's hypotheses over the evaluator's
behavioral diagnostics. The score ladder is shaped so composable partial
capabilities stay viable for branch preservation and recombination.
