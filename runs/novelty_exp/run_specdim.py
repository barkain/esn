"""Test the spectral fix: force a small fixed spectral_dim so gamma=d/H_t drops
below the undersampled gate and the BBP/MP detector can finally engage on 4o-mini.

Usage: run_specdim.py <spectral_dim|off> <seed> <gens>
  spectral_dim = integer (e.g. 8) -> novelty ON with that working dim
  "off"                          -> novelty OFF baseline

Reports best score + spectral engagement stats: how many generations detected
spikes, and how many candidates got a live (non-zero) N_sp. If the fix works we
should see spikes>0 and N_sp>0 once the bank clears ~30 hypotheses.
"""
import os
import sys
import json

os.environ["OPENAI_API_KEY"] = (
    os.environ.get("OPENAI_API_KEY_ESN") or os.environ.get("OPENAI_API_KEY", "")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "h2h_bf"))

import esn  # noqa: E402
import esn.api as api  # noqa: E402
from esn.core import novelty as novelty_mod  # noqa: E402
from esn.core import spectral as spectral_mod  # noqa: E402
if os.environ.get("DOMAIN") == "nz":
    from biasfree_nz import biasfree_nz_domain as biasfree_domain  # noqa: E402
else:
    from biasfree import biasfree_domain  # noqa: E402

arg = sys.argv[1] if len(sys.argv) > 1 else "8"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
gens = int(sys.argv[3]) if len(sys.argv) > 3 else 20
batch = int(sys.argv[4]) if len(sys.argv) > 4 else 4
nov = arg != "off"
spectral_dim = int(arg) if nov else None

# --- Neutralize the non-upstream PARENT_QUALITY_FLOOR_RATIO gate so the engine
# behaves like clean upstream (floor=0 -> gate passes everything). Set via env
# NEUTRALIZE_GATE=1 (default ON here for the clean comparison). ---
if os.environ.get("NEUTRALIZE_GATE", "1") == "1":
    from esn.engine import engine as _eng
    _eng.PARENT_QUALITY_FLOOR_RATIO = 0.0
    print(f"GATE_NEUTRALIZED floor={_eng.PARENT_QUALITY_FLOOR_RATIO}", flush=True)

# --- Strip the mutator's anti-optimization constraint (suspected prompt bias
# steering 4o-mini toward uniform greedy grids). Set STRIP_ANTIOPT=1. ---
if os.environ.get("STRIP_ANTIOPT") == "1":
    import re as _re
    from esn.engine import mutator as _mut
    _AO = _re.compile(r"CRITICAL RUNTIME CONSTRAINT.*?greedy algorithms with early stopping\.\n", _re.S)
    _orig_sp = _mut.LLMMutator._build_system_prompt
    def _stripped_sp(self, style):
        return _AO.sub("Use bounded loops so the program finishes within the time limit.\n", _orig_sp(self, style))
    _mut.LLMMutator._build_system_prompt = _stripped_sp
    print("ANTIOPT_STRIPPED", flush=True)

# --- OpenEvolve-spirit mutator prompt (encourages optimization/varied sizes/
# rewrites, no forbidding language, no specific tool names). Set OPENEVOLVE_PROMPT=1. ---
if os.environ.get("OPENEVOLVE_PROMPT") == "1":
    import oe_prompt
    oe_prompt.install()
    print("OPENEVOLVE_PROMPT installed", flush=True)

# --- Force spectral_dim before the compressor is constructed ---
if nov:
    def patched_stack(seed, spectral_threshold_mode="empirical"):
        from esn.core.spectral_models import ESNConfig
        from esn.core.knowledge import KnowledgeIntegration
        from esn.core.novelty import NoveltyComputer
        from esn.core.embeddings import SentenceTransformerEmbedder
        config = ESNConfig()
        config.spectral_threshold_mode = spectral_threshold_mode
        config.spectral_dim = spectral_dim
        embedder = SentenceTransformerEmbedder(config.embedding_model)
        knowledge = KnowledgeIntegration(config=config, embedder=embedder)
        nc = NoveltyComputer(knowledge, config=config, seed=seed)
        print(f"PATCHED spectral_dim={config.spectral_dim} compressor_target={nc._compressor.target_dim}", flush=True)
        return knowledge, nc, config

    api._build_novelty_stack = patched_stack

# --- Engagement stats ---
stats = {"gen_spikes": [], "sp_real": 0, "sp_zero": 0}
_orig_guid = spectral_mod.compute_spectral_guidance


def traced_guid(**kw):
    stats["gen_spikes"].append((len(kw.get("hypotheses") or []), kw.get("spike_count"), kw.get("undersampled")))
    return _orig_guid(**kw)


spectral_mod.compute_spectral_guidance = traced_guid

_orig_compute = novelty_mod.NoveltyComputer.compute


def traced_compute(self, *a, **k):
    ep, sp, unified = _orig_compute(self, *a, **k)
    if sp and sp > 0:
        stats["sp_real"] += 1
    else:
        stats["sp_zero"] += 1
    return ep, sp, unified


novelty_mod.NoveltyComputer.compute = traced_compute

# --- Count actual candidate evaluations (to verify matched budget) ---
from esn.engine.engine import ESNEngine as _ENG
_evalcount = {"n": 0}
_orig_po = _ENG._process_outcome
def _counted_po(self, outcome):
    _evalcount["n"] += 1
    return _orig_po(self, outcome)
_ENG._process_outcome = _counted_po

domain = biasfree_domain()
GEN_MODEL = os.environ.get("GEN_MODEL", "gpt-4o-mini")
mutator = esn.make_llm_mutator(domain, model=GEN_MODEL)
analyzer = esn.make_analyzer(model=GEN_MODEL) if nov else None
predictor = esn.make_predictor(model=GEN_MODEL) if nov else None
r = esn.run(
    domain, mutator=mutator, analyzer=analyzer, predictor=predictor,
    generations=gens, batch_size=batch, seed=seed, enable_recombination=True,
)
gens_with_spikes = sum(1 for (_h, s, _u) in stats["gen_spikes"] if s and s > 0)
gens_engaged = sum(1 for (_h, _s, u) in stats["gen_spikes"] if u is False)
out = {
    "spectral_dim": spectral_dim, "arm": arg, "seed": seed, "gens": gens,
    "best_score": float(r.best_score),
    "gens_with_spikes": gens_with_spikes,
    "gens_not_undersampled": gens_engaged,
    "sp_real": stats["sp_real"], "sp_zero": stats["sp_zero"],
    "n_evals": _evalcount["n"],
    "trajectory": [round(float(h.get("best_score", 0)), 4) for h in (r.history or [])],
    "bank_spike_under": stats["gen_spikes"],
}
print("SPECDIM_RESULT " + json.dumps(out), flush=True)
