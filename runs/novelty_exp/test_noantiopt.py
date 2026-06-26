"""Prime-suspect test: does REMOVING the mutator's anti-optimization constraint
('avoid multi-phase optimization, prefer greedy single-pass') let 4o-mini exceed
the 2.167 uniform-grid ceiling on the nz task? Monkeypatch _build_system_prompt
to strip that block; everything else identical (clean engine, nz, novelty-off)."""
import os, sys, re
os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
os.environ["DOMAIN"]="nz"
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","h2h_bf"))
import esn
from esn.engine import engine as _eng
_eng.PARENT_QUALITY_FLOOR_RATIO=0.0
from esn.engine import mutator as mut
from biasfree_nz import biasfree_nz_domain

_orig=mut.LLMMutator._build_system_prompt
ANTIOPT_RE=re.compile(r"CRITICAL RUNTIME CONSTRAINT.*?greedy algorithms with early stopping\.\n", re.S)
def patched(self, style):
    p=_orig(self, style)
    p2=ANTIOPT_RE.sub("Use bounded loops so the program finishes within the time limit.\n", p)
    return p2
mut.LLMMutator._build_system_prompt=patched
# confirm the strip worked once
sample=patched(mut.LLMMutator.__new__(mut.LLMMutator).__class__.__dict__ and None or None, "refine") if False else None

d=biasfree_nz_domain()
seed=int(sys.argv[1]) if len(sys.argv)>1 else 42
gens=int(sys.argv[2]) if len(sys.argv)>2 else 20
r=esn.run(d, mutator=esn.make_llm_mutator(d, model="gpt-4o-mini"),
          analyzer=None,predictor=None,generations=gens,batch_size=4,seed=seed,enable_recombination=True)
print(f"NOANTIOPT seed={seed} best={r.best_score:.4f} (grid line=2.167)", flush=True)
