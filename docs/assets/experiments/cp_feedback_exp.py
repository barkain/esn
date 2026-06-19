"""Feedback-learning experiment on the VALIDATED cp harness.

Tests the operator team's #1 idea: ground the direction in the REAL objective
(score + why-it-fell-short) instead of the model's intuition (which misfires —
see self-diagnosis 0/96). Feedback describes the SCORING MECHANICS (wasted
zero-radius circles; radius capped by nearest-neighbor+wall) — it never names a
strategy ("grid"/"hex"), per the critic's no-answer-leak control.

Arms (N=96), HUMAN validates the harness (~7-9%):
  FEEDBACK : round1 neutral -> objective-grounded feedback -> round2 revise (2 calls)
  PLACEBO  : round1 neutral -> generic "try different" feedback -> round2 (2 calls)
  HUMAN    : F1 hint, 1 call (ceiling/validation)
Decision: FEEDBACK cracks it iff > PLACEBO and approaches/beats HUMAN.
"""
import os, re, json
import numpy as np
from scipy.optimize import linprog
from scipy.stats import fisher_exact
import openai

WEAK = "gpt-4o-mini"
N = int(os.environ.get("N", "96"))
THRESH, NC, PARENT = 2.30, 26, 2.229
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ART = open("/tmp/cp_feedback_artifacts.jsonl", "w")

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

def score_candidate(comp):
    """Return (lp_score, nonzero, morph, valid)."""
    prog = SCAFFOLD.replace("{C}", comp)
    c = extract(prog)
    if c is None or len(c) < NC: return None
    c = c[:NC]; r = lp_radii(c)
    return float(r.sum()), int((r > 1e-3).sum()), morph(c)

def judge(comp, arm):
    rec = {"arm": arm}
    sc = score_candidate(comp)
    if sc is None:
        rec["hard"] = False; ART.write(json.dumps(rec) + "\n"); return False
    s, nz, m = sc
    hard = (s >= THRESH and nz == NC and m == "other" and s > PARENT)
    rec.update(lp=round(s, 4), nonzero=nz, morph=m, hard=bool(hard))
    ART.write(json.dumps(rec) + "\n")
    return hard

SYS = "You are an expert in geometric optimization. Return ONLY a Python function, no markdown, no commentary."
def code_of(t):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", t, re.DOTALL)
    return (m.group(1) if m else t).strip()
def call(usr, mt=1500):
    t = client.chat.completions.create(model=WEAK, max_tokens=mt,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": usr}]
        ).choices[0].message.content or ""
    return code_of(t)

NEUTRAL_USER = ("# Task\nWrite a function `place_centers()` returning a numpy array of 26 (x,y) coordinates inside the "
    "unit square so non-overlapping circles centered there have the LARGEST possible sum of radii.\n\n"
    f"# Current solution for reference\n{RING}\n\nReturn ONLY the def place_centers(): function.")
HUMAN_USER = ("# Task\nWrite a function `place_centers()` returning a numpy array of 26 (x,y) coordinates inside the "
    "unit square so non-overlapping circles centered there have the LARGEST possible sum of radii.\n\n"
    "# Spectral guidance\nThe CURRENT arrangement is a concentric-ring layout that has plateaued — produce a CATEGORICALLY DIFFERENT spatial arrangement.\n\n"
    f"# Current solution for reference\n{RING}\n\nReturn ONLY the def place_centers(): function.")

def grounded_feedback(sc):
    """Objective-grounded, NON-strategy-leaking feedback from the real scoring."""
    if sc is None:
        return "Your previous code did not produce 26 valid circle centers. Produce exactly 26 centers inside the unit square."
    s, nz, m = sc
    parts = [f"Your previous arrangement scored {s:.2f} (you must exceed {PARENT:.2f} to improve)."]
    if nz < NC:
        parts.append(f"{NC-nz} of 26 circles ended with near-zero radius — they are wasted and contribute nothing to the total.")
    parts.append("Each circle's radius is capped by the distance to its nearest neighbor and to the walls. "
                 "To raise the TOTAL sum, reposition the centers so that more circles have room to grow large, "
                 "rather than many circles crammed close together.")
    return " ".join(parts)

PLACEBO_FEEDBACK = "Your previous arrangement was not good enough. Try a different arrangement that scores higher."

def round2_user(prev_code, feedback):
    return ("# Task\nWrite a function `place_centers()` returning 26 (x,y) coordinates inside the unit square so "
        "non-overlapping circles have the LARGEST possible sum of radii.\n\n"
        f"# Your previous attempt\n{prev_code}\n\n# Feedback\n{feedback}\n\n"
        "Now write an IMPROVED def place_centers(). Return ONLY the function.")

def run_feedback(name, grounded, n, log):
    hits = 0
    for i in range(n):
        try:
            r1 = call(NEUTRAL_USER)
            if "def place_centers" not in r1:
                judge("", name); continue
            sc = score_candidate(r1)
            fb = grounded_feedback(sc) if grounded else PLACEBO_FEEDBACK
            if i < 8: log.append(fb)
            r2 = call(round2_user(r1, fb))
            if "def place_centers" in r2 and judge(r2, name): hits += 1
            else: judge(r2 if "def place_centers" in r2 else "", name)
        except Exception:
            judge("", name)
    return hits

def run_human(name, n):
    hits = 0
    for _ in range(n):
        try:
            c = call(HUMAN_USER)
            if "def place_centers" in c and judge(c, name): hits += 1
            else: judge("", name)
        except Exception:
            judge("", name)
    return hits

if __name__ == "__main__":
    out = open("/tmp/cp_feedback_results.txt", "w")
    def P(s): print(s, flush=True); out.write(s + "\n"); out.flush()
    P(f"parent={PARENT} THRESH={THRESH} | N={N} | hardened=LP>=2.30 & 26nz & non-ring & >parent")
    fb_log = []
    h_fb = run_feedback("FEEDBACK", True, N, fb_log)
    h_pl = run_feedback("PLACEBO", False, N, [])
    h_hu = run_human("HUMAN", N)
    def fish(h, base): return fisher_exact([[h, N-h], [base, N-base]], alternative="greater")[1]
    P(f"FEEDBACK hardened={h_fb:3}/{N} = {h_fb/N*100:.1f}%  (Fisher vs PLACEBO p={fish(h_fb,h_pl):.3f})")
    P(f"PLACEBO  hardened={h_pl:3}/{N} = {h_pl/N*100:.1f}%  [2-call control]")
    P(f"HUMAN    hardened={h_hu:3}/{N} = {h_hu/N*100:.1f}%  [ceiling/validation: must be ~7-9%]")
    P(f"\nVALIDATION: HUMAN {'OK' if h_hu/N>=0.04 else 'FAILED -> harness suspect'}")
    if h_hu/N >= 0.04:
        if h_fb > h_pl and fish(h_fb, h_pl) < 0.05:
            P("VERDICT: FEEDBACK beats PLACEBO significantly -> objective-grounded feedback UNLOCKS escape. THE FIX.")
        elif h_fb >= h_hu*0.7:
            P("VERDICT: FEEDBACK ~ HUMAN but ~ PLACEBO too -> gains are from the 2nd attempt, not the specific feedback.")
        else:
            P("VERDICT: FEEDBACK does not beat PLACEBO and is below HUMAN -> grounded feedback does not unlock escape here.")
    P("\n--- 8 sample grounded-feedback strings ---")
    for s in fb_log: P(f"  • {s}")
    P("\nDONE")
    ART.close(); out.close()
