"""Run the bestofN config (gens=1, batch=80, gate off, novelty off) and save
EVERY candidate's (score, code). Then inspect the top scorers: do they reach
high scores via a runtime OPTIMIZER embedded in the generated program (scipy /
iterative refinement loops), or via a static construction? This tests whether
"4o-mini hit 2.5 in one batch" is the LLM guessing vs a program that optimizes
at eval time.
"""
import os, sys, json, re
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h2h_bf"))

import esn
from esn.engine import engine as _eng
_eng.PARENT_QUALITY_FLOOR_RATIO = 0.0
from esn.engine.engine import ESNEngine
from biasfree import biasfree_domain

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 45
cands = []
_orig = ESNEngine._process_outcome
def hook(self, outcome):
    rec = _orig(self, outcome)
    cands.append((float(rec.score or 0.0), outcome.style, outcome.new_code or ""))
    return rec
ESNEngine._process_outcome = hook

domain = biasfree_domain()
r = esn.run(domain, mutator=esn.make_llm_mutator(domain, model="gpt-4o-mini"),
            analyzer=None, predictor=None, generations=1, batch_size=80,
            seed=seed, enable_recombination=False)

cands.sort(reverse=True, key=lambda x: x[0])
print(f"RUN seed={seed} best={r.best_score:.4f} n_candidates={len(cands)}")
def feats(code):
    return {
        "scipy": bool(re.search(r"scipy|minimize|linprog|differential_evolution", code)),
        "anneal/gd": bool(re.search(r"anneal|gradient|temperature|perturb", code, re.I)),
        "iter_loops": len(re.findall(r"for .*range\(|while ", code)),
        "lines": len(code.splitlines()),
    }
print("\n=== TOP 5 candidates (score, style, features) ===")
for s, style, code in cands[:5]:
    print(f"score={s:.4f} style={style} {feats(code)}")
# dump the single best candidate's full code
best_code = cands[0][2]
open(os.path.join(os.path.dirname(__file__), f"best_cand_seed{seed}.py"), "w").write(best_code)
print(f"\n=== BEST CANDIDATE (score={cands[0][0]:.4f}) full code ===")
print(best_code)
