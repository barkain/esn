import os, sys, re, numpy as np
os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
os.environ["DOMAIN"]="nz"
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","h2h_bf"))
sys.path.insert(0, os.path.dirname(__file__))
import esn, oe_prompt
oe_prompt.install()
from esn.engine import engine as _eng
_eng.PARENT_QUALITY_FLOOR_RATIO=0.0
from esn.engine.engine import ESNEngine
from biasfree_nz import biasfree_nz_domain
from circle_packing.domain import evaluate_circle_packing_artifact as ev
best=[None]
_o=ESNEngine._process_outcome
def hook(self,o):
    r=_o(self,o)
    if (r.score or 0)>(best[0][0] if best[0] else 0): best[0]=(float(r.score), o.new_code or "")
    return r
ESNEngine._process_outcome=hook
d=biasfree_nz_domain()
esn.run(d, mutator=esn.make_llm_mutator(d, model="gpt-4o-mini"), analyzer=None,predictor=None,
        generations=1, batch_size=80, seed=42, enable_recombination=False)
s,code=best[0]
# independent audit: run it, recompute, validate from scratch
ns={}; exec(code, ns); fn=ns.get("solve") or ns.get("construct_packing")
out=fn(); c=np.asarray(out[0],float); rad=np.asarray(out[1],float)
res=ev((c,rad))
# independent overlap/bounds/positivity check
ok=True; reason="valid"
if c.shape!=(26,2) or rad.shape!=(26,): ok=False; reason="shape"
if rad.min()<=0: ok=False; reason=f"nonpositive r (min={rad.min():.4g})"
for i in range(26):
    x,y=c[i]
    if x-rad[i]<-1e-6 or y-rad[i]<-1e-6 or x+rad[i]>1+1e-6 or y+rad[i]>1+1e-6: ok=False; reason=f"oob circle {i}"
worst=0
for i in range(26):
  for j in range(i+1,26):
    ov=rad[i]+rad[j]-np.hypot(*(c[i]-c[j])); worst=max(worst,ov)
    if ov>1e-6: ok=False; reason=f"overlap {i},{j}={ov:.4g}"
print(f"CAPTURED best={s:.5f}  evaluator: score={res.score:.5f} success={res.success}")
print(f"INDEPENDENT AUDIT: valid={ok} ({reason})  worst_overlap={worst:.2e}  sum_radii={rad.sum():.5f}")
print(f"radii: min={rad.min():.4f} max={rad.max():.4f} std={rad.std():.4f} (varied if std>0)  n_distinct={len(set(np.round(rad,3)))}")
print(f"uses scipy/optimize: {bool(re.search(r'scipy|minimize|optimize|SLSQP',code))}  lines={len(code.splitlines())}")
print("==== CODE ===="); print(code)
