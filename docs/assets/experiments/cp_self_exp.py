"""Self-generated steer experiment on the VALIDATED cp_a3_run.py machinery.

Only the STEER SOURCE varies; the implementation call (SYS, user(steer),
max_tokens=1500), SCAFFOLD, RING, LP+hardened judge are IDENTICAL to the
validated harness (which gives HUMAN=18/192=9.4%). So the HUMAN arm here MUST
reproduce ~9% — that validates the measuring stick before we trust SELF.

Arms (N=96): SELF (per-sample planner steer) | BLAND (floor) | HUMAN (ceiling).
Decision: SELF cracks it iff SELF >> BLAND and approaches HUMAN.
"""
import os, re, json, statistics
import numpy as np
from scipy.optimize import linprog
from scipy.stats import fisher_exact
import openai

WEAK = "gpt-4o-mini"
N = int(os.environ.get("N", "96"))
THRESH, NC, PARENT = 2.30, 26, 2.229
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ART = open("/tmp/cp_self_artifacts.jsonl", "w")

# ---- VALIDATED machinery (verbatim from /tmp/cp_a3_run.py) ----
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

def impl(steer):
    t = client.chat.completions.create(model=WEAK, max_tokens=1500,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user(steer)}]
        ).choices[0].message.content or ""
    return code_of(t)

# ---- steers ----
BLAND = "The reference arrangement below has stopped improving (it has converged). Return a DIFFERENT arrangement rather than reproducing the reference."
HUMAN = "The CURRENT arrangement is a concentric-ring layout that has plateaued — produce a CATEGORICALLY DIFFERENT spatial arrangement."

PLANNER_SYS = "You are an expert in geometric optimization. Output ONLY JSON."
PLANNER_USER = (
    "A circle-packing program places 26 circles to maximize the sum of radii. The CURRENT solution scores 2.229 and is:\n"
    f"{RING}\n\n"
    "Diagnose why this plateaued and emit a one-sentence positive steer for a BETTER arrangement.\n"
    "RULES: the steer must name a CONCRETE constructive geometry family (what TO build). "
    "It must NOT use generic words (novel/different/diverge) and must NOT say 'avoid'. "
    'Return JSON: {"failure_mode": "...", "one_sentence_positive_steer": "..."}')

def planner_steer():
    t = client.chat.completions.create(model=WEAK, max_tokens=200,
        messages=[{"role": "system", "content": PLANNER_SYS}, {"role": "user", "content": PLANNER_USER}]
        ).choices[0].message.content or ""
    raw = t
    if "```" in t:
        t = t.split("```json")[-1].split("```")[1] if "```json" in t else t.split("```")[1]
    try:
        j = json.loads(t.strip())
        return j.get("one_sentence_positive_steer", "")[:300], raw
    except Exception:
        return "", raw

def run_fixed(name, steer, n):
    hits = 0
    for _ in range(n):
        try:
            comp = impl(steer)
            if "def place_centers" in comp and judge(SCAFFOLD.replace("{C}", comp), name):
                hits += 1
        except Exception:
            judge("", name)
    return hits

def run_self(name, n, log):
    hits = 0
    for i in range(n):
        try:
            steer, raw = planner_steer()
            if i < 12:
                log.append(steer)
            if not steer:
                judge("", name); continue
            comp = impl(steer)
            if "def place_centers" in comp and judge(SCAFFOLD.replace("{C}", comp), name):
                hits += 1
        except Exception:
            judge("", name)
    return hits

if __name__ == "__main__":
    out = open("/tmp/cp_self_results.txt", "w")
    def P(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    P(f"parent={PARENT} THRESH={THRESH} hardened=LP>=2.30 & 26nz & non-ring & >parent | N={N}")
    steers_log = []
    h_self = run_self("SELF", N, steers_log)
    h_bland = run_fixed("BLAND", BLAND, N)
    h_human = run_fixed("HUMAN", HUMAN, N)
    def fish(h, base):
        return fisher_exact([[h, N - h], [base, N - base]], alternative="greater")[1]
    P(f"SELF   hardened={h_self:3}/{N} = {h_self/N*100:.1f}%  (Fisher vs BLAND p={fish(h_self,h_bland):.3f})")
    P(f"BLAND  hardened={h_bland:3}/{N} = {h_bland/N*100:.1f}%  [floor]")
    P(f"HUMAN  hardened={h_human:3}/{N} = {h_human/N*100:.1f}%  [ceiling/validation: must be ~9%]")
    P(f"\nVALIDATION: HUMAN {'OK (>=4%)' if h_human/N>=0.04 else 'FAILED <4% -> harness suspect'}")
    if h_human/N >= 0.04:
        if h_self > h_bland and fish(h_self, h_bland) < 0.05:
            P("VERDICT: SELF beats BLAND significantly -> weak model CAN self-author useful direction.")
        else:
            P("VERDICT: SELF ~= BLAND -> self-diagnosis does NOT unlock escape (weak model can't author effective direction).")
    P("\n--- 12 sample SELF steers ---")
    for s in steers_log: P(f"  • {s}")
    P("\nDONE")
    ART.close(); out.close()
