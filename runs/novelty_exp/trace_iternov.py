import os, sys, re, hashlib, numpy as np
os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
os.environ.update(DOMAIN="nz", NZ_TIMEOUT="90", NZ_SEED=os.path.join(os.path.dirname(__file__),"..","h2h_bf","scipy_seed.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","h2h_bf")); sys.path.insert(0, os.path.dirname(__file__))
import esn, oe_prompt; oe_prompt.install()
from esn.engine import engine as _eng; _eng.PARENT_QUALITY_FLOOR_RATIO=0.0
from esn.engine.engine import ESNEngine
from esn.core import spectral as spec
from biasfree_nz import biasfree_nz_domain
# spectral trace
guid=[]
_og=spec.compute_spectral_guidance
def tg(**k):
    guid.append((len(k.get('hypotheses') or []), k.get('spike_count'), k.get('undersampled'))); return _og(**k)
spec.compute_spectral_guidance=tg
# program capture
progs={}
_op=ESNEngine._process_outcome
def hook(self,o):
    r=_op(self,o); code=o.new_code or ""
    h=hashlib.md5(code.encode()).hexdigest()[:8]
    if h not in progs:
        scipy=bool(re.search(r'scipy|minimize|optimize',code))
        progs[h]=(round(float(r.score or 0),3), scipy, len(code.splitlines()), code[:0])
    return r
ESNEngine._process_outcome=hook
d=biasfree_nz_domain()
r=esn.run(d, mutator=esn.make_llm_mutator(d, model="gpt-4o-mini"),
          analyzer=esn.make_analyzer(model="gpt-4o-mini"), predictor=esn.make_predictor(model="gpt-4o-mini"),
          generations=8, batch_size=4, seed=42, enable_recombination=True)
import esn.api as api
# spectral_dim default is 8 in fix branch? force via patch like run_specdim — but here use default; report gamma via guid
print(f"TRACE_DONE best={r.best_score:.4f}")
print(f"unique_programs={len(progs)}  scipy_programs={sum(1 for v in progs.values() if v[1])}/{len(progs)}")
print("SPECTRAL per-gen (H_t, spike_count, undersampled):")
for g in guid: print("  ",g)
print("program scores+scipy:", sorted([(v[0],v[1]) for v in progs.values()], reverse=True)[:12])
