import os, sys, importlib.util, numpy as np
os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","h2h_bf"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","..","examples"))
import esn
from esn.engine.engine import ESNEngine
from biasfree import biasfree_domain
from circle_packing.domain import evaluate_circle_packing_artifact as ev
hits=[]
_o=ESNEngine._process_outcome
def hook(self,outcome):
    r=_o(self,outcome)
    if (r.score or 0)>=2.49: hits.append(outcome.new_code or "")
    return r
ESNEngine._process_outcome=hook
domain=biasfree_domain()
for seed in [43,44,45,42,47,48,49]:
    hits.clear()
    res=esn.run(domain, mutator=esn.make_llm_mutator(domain, model="gpt-4o-mini"),
                analyzer=None, predictor=None, generations=2, batch_size=40, seed=seed, enable_recombination=False)
    if hits:
        code=hits[0]
        ns={}; exec(code, ns)
        fn=ns.get("solve") or ns.get("construct_packing")
        out=fn(); c=np.asarray(out[0],float); rad=np.asarray(out[1],float)
        r=ev((c,rad))
        print(f"seed={seed} score={r.score:.10f} success={r.success}", flush=True)
        print(f"  n_circles={len(rad)} nonzero_radii={int((rad>1e-9).sum())} radii_unique={sorted(set(np.round(rad,4)))[:8]}", flush=True)
        print(f"  radii: {np.round(rad,4).tolist()}", flush=True)
        print("  ---- CODE (first 40 lines) ----", flush=True)
        print("\n".join(code.splitlines()[:40]), flush=True)
        break
    else:
        print(f"seed={seed}: no 2.5 this draw (best={res.best_score:.4f})", flush=True)
