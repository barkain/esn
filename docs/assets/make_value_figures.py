# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Render the ESN value figures from captured real-run data (no LLM needed).

Reads ``data/run_on.json`` (and optionally ``data/run_off.json``) — captured by
``capture_run.py`` from real ``circle_packing`` runs — and writes three PNGs next
to this script. Pure matplotlib, fully reproducible from the committed data:

    uv run --extra novelty python docs/assets/make_value_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ROUTE_STYLE = {
    "elite": ("#d4a017", "o", "elite archive (near-best)"),
    "frontier": ("#1f77b4", "o", "novelty frontier (viable, below-best)"),
    "not_retained": ("#bbbbbb", "o", "not retained"),
    "failed": ("#d62728", "x", "failed (success=False)"),
}


def _load(mode):
    p = DATA / f"run_{mode}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_frontier_survival(on):
    cands, gens = on["candidates"], on["generations"]
    by = [g["best"] for g in gens]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.step(
        [g["gen"] for g in gens],
        by,
        where="post",
        color="black",
        lw=2.2,
        label="running best (fitness wins)",
        zorder=5,
    )
    for route, (color, marker, label) in ROUTE_STYLE.items():
        xs = [c["gen"] for c in cands if c["route"] == route]
        ys = [c["score"] for c in cands if c["route"] == route]
        if xs:
            kw = {"edgecolors": "white", "linewidths": 0.5} if marker == "o" else {}
            ax.scatter(
                xs,
                ys,
                c=color,
                marker=marker,
                s=46,
                alpha=0.85,
                label=label,
                zorder=3,
                **kw,
            )
    # Punchline (real lineage): the run's best candidate descended from a
    # below-best candidate that only survived because the novelty frontier kept it.
    by_id = {c["id"]: c for c in cands}
    best_c = max((c for c in cands if c["success"]), key=lambda c: c["score"])
    par = by_id.get(best_c["parent"])
    if par and par["route"] == "frontier":
        ax.scatter(
            [par["gen"]],
            [par["score"]],
            s=170,
            facecolors="none",
            edgecolors="#1f77b4",
            lw=2.0,
            zorder=6,
        )
        ax.annotate(
            "",
            xy=(best_c["gen"], best_c["score"]),
            xytext=(par["gen"], par["score"]),
            arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.2),
            zorder=6,
        )
        ax.annotate(
            f"below-best ({par['score']:.2f}) at gen {par['gen']} — a greedy\n"
            f"keep-the-best loop discards it. The novelty\n"
            f"frontier keeps it; it becomes the parent of\n"
            f"the run's best ({best_c['score']:.2f}).",
            xy=(par["gen"], par["score"]),
            xytext=(4.8, 0.42),
            fontsize=8.6,
            color="#10558a",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#1f77b4", alpha=0.92),
            zorder=7,
        )
    ax.set_xlabel("generation")
    ax.set_ylabel("score  (sum of radii — higher is better)")
    ax.set_title(
        "ESN: fitness crowns the champion, novelty keeps viable alternatives alive\n"
        "circle_packing · one real run (Claude Haiku, seed 42)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "frontier-survival-circle-packing.png", dpi=150)
    print("wrote frontier-survival-circle-packing.png")


def fig_spectral_gate(on):
    gens = on["generations"]
    x = [g["gen"] for g in gens]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )
    ax1.plot(x, [g["best"] for g in gens], color="black", lw=2, marker="o", ms=3)
    ax1.set_ylabel("running best")
    ax1.set_title(
        "Spectral novelty (N_sp) activates conservatively — only after stable structure forms\n"
        "circle_packing · one real run with the [novelty] embedder",
        fontsize=11,
    )
    ax1.grid(True, alpha=0.25)
    ax2.bar(x, [g["spikes"] for g in gens], color="#9ecae1")
    ax2.set_ylabel("spikes", color="#3182bd")
    ax2.set_xlabel("generation")
    axg = ax2.twinx()
    gmax = max((g["gamma"] for g in gens), default=0.0)
    axg.plot(x, [g["gamma"] for g in gens], color="#e6550d", lw=2.2, marker="o", ms=3)
    axg.set_ylabel("γ  (0 = epistemic-only)", color="#e6550d")
    axg.set_ylim(0, max(0.3, gmax * 1.2 + 1e-6))
    act = next((g["gen"] for g in gens if g["gamma"] > 0), None)
    if act is not None:
        axg.axvline(act, color="#e6550d", ls="--", lw=1, alpha=0.6)
        axg.annotate(
            f"γ>0 at gen {act}\n(spikes persisted ≥ 3 gens)",
            xy=(act, 0.01),
            xytext=(act - 5.0, gmax * 0.55 + 0.02),
            fontsize=8.5,
            color="#e6550d",
        )
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(HERE / "spectral-gate-circle-packing.png", dpi=150)
    print("wrote spectral-gate-circle-packing.png")


if __name__ == "__main__":
    on = _load("on")
    assert on is not None, "missing data/run_on.json"
    fig_frontier_survival(on)
    fig_spectral_gate(on)
