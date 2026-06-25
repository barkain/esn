import os, sys, re
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY","")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h2h_bf"))
import esn
from esn.engine import engine as _eng
_eng.PARENT_QUALITY_FLOOR_RATIO = 0.0
from esn.engine.engine import ESNEngine
from biasfree import biasfree_domain
hi = []
_orig = ESNEngine._process_outcome
def hook(self, outcome):
    rec = _orig(self, outcome)
    s = float(rec.score or 0.0)
    if s >= 2.3: hi.append((s, outcome.style, outcome.new_code or ""))
    return rec
ESNEngine._process_outcome = hook
domain = biasfree_domain()
for seed in [42,44,46,47]:
    hi.clear()
    r = esn.run(domain, mutator=esn.make_llm_mutator(domain, model="gpt-4o-mini"),
                analyzer=None, predictor=None, generations=1, batch_size=80,
                seed=seed, enable_recombination=False)
    print(f"=== seed={seed} best={r.best_score:.4f} #cands>=2.3: {len(hi)} ===", flush=True)
    if hi:
        hi.sort(reverse=True)
        s,style,code = hi[0]
        f={"scipy":bool(re.search(r"scipy|minimize|linprog|differential_evolution",code)),
           "anneal/gd":bool(re.search(r"anneal|gradient|temperature|perturb",code,re.I)),
           "loops":len(re.findall(r"for .*range\(|while ",code)),"lines":len(code.splitlines())}
        print(f"  TOP score={s:.4f} style={style} feats={f}", flush=True)
        print("  ---- CODE ----", flush=True)
        print(code, flush=True)
        break
