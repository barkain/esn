import os, sys
os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
os.environ["DOMAIN"]="nz"
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","h2h_bf"))
import esn
from esn.engine.engine import ESNEngine
from biasfree_nz import biasfree_nz_domain
scores=[]
_o=ESNEngine._process_outcome
def hook(self,o):
    r=_o(self,o); scores.append((float(r.score or 0),r.success)); return r
ESNEngine._process_outcome=hook
d=biasfree_nz_domain()
for seed in [42,43,44]:
    scores.clear()
    r=esn.run(d, mutator=esn.make_llm_mutator(d, model="gpt-4o-mini"),
              analyzer=None,predictor=None,generations=1,batch_size=80,seed=seed,enable_recombination=False)
    succ=[s for s,ok in scores if ok and s>0]
    fail=sum(1 for s,ok in scores if not ok or s==0)
    top=sorted(succ,reverse=True)[:5]
    print(f"seed={seed} NZ best={r.best_score:.4f} | valid={len(succ)}/{len(scores)} failed={fail} | top5={['%.3f'%x for x in top]}", flush=True)
