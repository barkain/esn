"""Step 4: A‴ generation ablation — does the REAL v3 spectral guidance unlock escape?

Component-splice operator, gpt-4o-mini, scored by order-free LP allocator +
hardening (LP>=2.30, 26 nonzero, non-ring, > parent 2.229). Each arm differs
ONLY in the "# Spectral guidance" block injected into the mutator prompt, mirroring
claude_agent_client.py:861 (full dict string). N=192 core arms (95% power vs zero
if true rate ~4.7%), N=96 diagnostic.

Arms:
  A'        no guidance ("(no spectral guidance provided)")
  A''       bland generic-diverge
  A3f       REAL guidance dict, n30 state ("Explore freely" + diagnostics)
  A3r       REAL guidance dict, n60 state (ring-list well/under-explored)
  A         human hint "ring plateaued -> categorically different spatial"
  A3r_avoid DIAGNOSTIC ONLY: ring-list manually translated to explicit anti-ring
"""
import os, re, json, statistics
import numpy as np
from scipy.optimize import linprog
from scipy.stats import fisher_exact
import openai

WEAK = "gpt-4o-mini"
N = int(os.environ.get("N", "192"))
NDIAG = int(os.environ.get("NDIAG", "96"))
THRESH, NC, PARENT = 2.30, 26, 2.229
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ART = open("/tmp/cp_a3_artifacts.jsonl", "w")
STEERS = json.load(open("/tmp/a3_steers.json"))

def lp_radii(c):
    c = np.clip(np.asarray(c, float).reshape(-1, 2), 0, 1); n = len(c)
    if n == 0: return np.zeros(0)
    wall = np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])
    rows, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.zeros(n); r[i] = 1; r[j] = 1; rows.append(r)
            b.append(float(np.hypot(*(c[i] - c[j]))))
    res = linprog(-np.ones(n), A_ub=np.array(rows), b_ub=np.array(b),
                  bounds=[(0, float(w)) for w in wall], method="highs")
    return np.maximum(res.x, 0.0) if res.success else np.zeros(n)

def morph(c):
    c = np.asarray(c, float).reshape(-1, 2)
    d = np.hypot(c[:, 0] - 0.5, c[:, 1] - 0.5)
    _, ct = np.unique(np.round(d / 0.05), return_counts=True)
    return "ring" if (np.sort(ct)[-3:].sum() >= 0.8 * len(c) and len(ct) <= 5) else "other"

def extract(prog):
    ns = {"np": np, "numpy": np, "__builtins__": __builtins__}
    try:
        exec(prog, ns)
        out = ns["place_centers"]() if "place_centers" in ns else ns["solve"]()
        return np.asarray(out[0] if isinstance(out, tuple) else out, float).reshape(-1, 2)
    except Exception:
        return None

SCAFFOLD = "import numpy as np\n{C}\ndef solve():\n    c=np.asarray(place_centers(),float).reshape(-1,2)[:26]\n    return c,np.zeros(len(c))\n"
RING = """def place_centers():
    c=np.zeros((26,2)); c[0]=[.5,.5]
    for i in range(8):
        a=2*np.pi*i/8; c[1+i]=[.5+.25*np.cos(a),.5+.25*np.sin(a)]
    for i in range(17):
        a=2*np.pi*i/17+np.pi/17; c[9+i]=[.5+.42*np.cos(a),.5+.42*np.sin(a)]
    return np.clip(c,.02,.98)"""

def judge(prog, arm):
    c = extract(prog)
    rec = {"arm": arm}
    if c is None or len(c) < NC:
        rec["hard"] = False; ART.write(json.dumps(rec) + "\n"); return False
    c = c[:NC]; r = lp_radii(c); s = float(r.sum()); nz = int((r > 1e-3).sum()); m = morph(c)
    hard = (s >= THRESH and nz == NC and m == "other" and s > PARENT)
    rec.update(lp=round(s, 4), nonzero=nz, morph=m, hard=bool(hard))
    ART.write(json.dumps(rec) + "\n")
    return hard

# steer blocks (the "# Spectral guidance" content)
BLAND = "The reference arrangement below has stopped improving (it has converged). Return a DIFFERENT arrangement rather than reproducing the reference."
HUMAN = "The CURRENT arrangement is a concentric-ring layout that has plateaued — produce a CATEGORICALLY DIFFERENT spatial arrangement."
AVOID = ("Spectral analysis indicates radial/concentric-ring structure is the well-explored, saturated direction. "
         "AVOID ring/radial/annular layouts entirely; use a fundamentally different spatial structure.")
ARMS = {
    "A'  no-guidance":    ("(no spectral guidance provided)", N),
    "A'' bland-diverge":  (BLAND, N),
    "A3f real-freely":    (STEERS["A3_freely"], N),
    "A3r real-ringlist":  (STEERS["A3_ring"], N),
    "A   human-hint":     (HUMAN, N),
    "A3r+avoid DIAG":     (AVOID, NDIAG),
}

SYS = "You are an expert in geometric optimization. Return ONLY a Python function, no markdown, no commentary."
def user(steer):
    return ("# Task\nWrite a function `place_centers()` returning a numpy array of 26 (x,y) coordinates inside the "
            "unit square, arranged so non-overlapping circles centered there have the LARGEST possible sum of radii.\n\n"
            f"# Spectral guidance\n{steer}\n\n"
            f"# Current solution for reference\n{RING}\n\n"
            "Return ONLY the def place_centers(): function.")
def code_of(t):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    return (m.group(1) if m else t).strip()

def run_arm(name, steer, n):
    hits = 0
    for _ in range(n):
        try:
            t = client.chat.completions.create(model=WEAK, max_tokens=1500,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user(steer)}]
                ).choices[0].message.content or ""
            comp = code_of(t)
            if "def place_centers" in comp and judge(SCAFFOLD.replace("{C}", comp), name):
                hits += 1
        except Exception:
            judge("", name)
    return hits

if __name__ == "__main__":
    print(f"parent LP={PARENT} | THRESH={THRESH} | hardened = LP>={THRESH} & 26-nonzero & non-ring & >parent")
    base = None
    results = {}
    for name, (steer, n) in ARMS.items():
        h = run_arm(name, steer, n)
        results[name] = (h, n)
        line = f"{name:20} N={n:3} | hardened={h:3}/{n} = {h/n*100:.1f}%"
        if base is None:
            base = (h, n)
        else:
            _, p = fisher_exact([[h, n - h], [base[0], base[1] - base[0]]], alternative="greater")
            line += f" | Fisher vs A' p={p:.3f}"
        print(line, flush=True)
    ART.close()
    print("DONE  artifacts=/tmp/cp_a3_artifacts.jsonl")
