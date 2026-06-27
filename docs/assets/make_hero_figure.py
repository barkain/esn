# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Render the README hero figure: evolution climbing to near-SOTA.

Eye-catching, single-panel summary of the budget-dependent circle-packing study
(``examples/circle_packing/experiments/``). Plots the two *real* heavy-evolution
runs (40 generations, gpt-4o-mini, fixed OpenEvolve-spirit prompt, seeds 42 & 43)
climbing in discrete steps from the scipy seed (~2.595) past the best-of-N
sampling plateau (~2.614) up to ≈ the AlphaEvolve SOTA (2.635).

Reads the committed trajectories from ``runs/novelty_exp/results_heavy.jsonl``
and writes ``evolution-climbs-to-sota.png`` next to this script. Pure matplotlib,
fully reproducible from committed data (no LLM call):

    uv run --with matplotlib python docs/assets/make_hero_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEAVY = REPO / "runs" / "novelty_exp" / "results_heavy.jsonl"

SOTA = 2.635  # AlphaEvolve published best for 26 circles

# Brand-ish palette: deep ink, two vivid climb lines, warm SOTA gold.
INK = "#11151c"
GRID = "#dfe3ea"
SAMPLING = "#7a8699"
GOLD = "#e8a317"
LINE_A = "#2f6df6"  # seed 42
LINE_B = "#10b3a3"  # seed 43


def _load_runs():
    """Return {'evolution': [(seed, trajectory)], 'sampling': [scores]}."""
    evo, samp = [], []
    for line in HEAVY.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["label"] == "iter_nov40":
            evo.append((d["seed"], d["trajectory"], d["gens_with_spikes"]))
        elif d["label"] == "bestof160":
            samp.append(d["best_score"])
    evo.sort(key=lambda r: r[0])
    return evo, samp


def main() -> None:
    evo, samp = _load_runs()
    sampling_plateau = sum(samp) / len(samp)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c4cad4",
            "axes.linewidth": 1.0,
        }
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    # --- headroom band: the gap evolution closes ------------------------------
    ax.axhspan(sampling_plateau, SOTA, color=GOLD, alpha=0.07, zorder=0)

    # --- reference lines ------------------------------------------------------
    ax.axhline(SOTA, color=GOLD, lw=2.6, zorder=2)
    ax.axhline(sampling_plateau, color=SAMPLING, lw=1.8, ls=(0, (6, 4)), zorder=2)

    # --- the two real climbs --------------------------------------------------
    styles = [(LINE_A, "seed 42", 2.628), (LINE_B, "seed 43", 2.632)]
    last_gen = 0
    for (seed, traj, spikes), (color, label, _) in zip(evo, styles):
        gens = list(range(1, len(traj) + 1))
        last_gen = max(last_gen, gens[-1])
        # start flat from the seed at gen 0
        xs = [0] + gens
        ys = [traj[0]] + list(traj)
        ax.step(
            xs,
            ys,
            where="post",
            color=color,
            lw=3.0,
            solid_capstyle="round",
            zorder=5,
            label=f"ESN evolution · {label}  ({spikes}/40 gens spectral-active)",
        )
        # dot every step up (a new best)
        prev = ys[0]
        for x, y in zip(xs, ys):
            if y > prev + 1e-6:
                ax.scatter([x], [y], s=44, color=color, edgecolor="white", lw=1.4, zorder=6)
            prev = max(prev, y)
        # end-value tag
        ax.annotate(
            f"{traj[-1]:.3f}",
            xy=(gens[-1], traj[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # seed marker
    seed_y = evo[0][1][0]
    ax.scatter([0], [seed_y], s=70, color=INK, zorder=7)
    ax.annotate(
        f"scipy seed\n{seed_y:.3f}",
        xy=(0, seed_y),
        xytext=(8, -26),
        textcoords="offset points",
        fontsize=9.5,
        color=INK,
    )

    # --- reference labels (on-plot) ------------------------------------------
    ax.annotate(
        f"AlphaEvolve SOTA  ·  {SOTA:.3f}",
        xy=(0.2, SOTA),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="#a9730a",
        path_effects=[withStroke(linewidth=3, foreground="white")],
    )
    ax.annotate(
        f"best-of-N sampling plateaus  ·  {sampling_plateau:.3f}",
        xy=(1.4, sampling_plateau),
        xytext=(0, -14),
        textcoords="offset points",
        ha="left",
        fontsize=10.5,
        color="#5b6675",
        path_effects=[withStroke(linewidth=3, foreground="white")],
    )
    ax.annotate(
        "headroom evolution\ncloses",
        xy=(last_gen * 0.5, (sampling_plateau + SOTA) / 2),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#a9730a",
        alpha=0.9,
        style="italic",
    )

    # --- frame ----------------------------------------------------------------
    ax.set_xlim(-0.6, last_gen + 4.2)
    ax.set_ylim(seed_y - 0.012, SOTA + 0.006)
    ax.set_xlabel("evolution generation", fontsize=11.5, color=INK)
    ax.set_ylabel("best packing score  (sum of 26 radii)", fontsize=11.5, color=INK)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(
        "Evolution climbs to state-of-the-art where sampling plateaus",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.set_title(
        "26-circle packing · gpt-4o-mini, identical fixed prompt · two real ESN runs "
        "(40 generations, spectral novelty on)",
        loc="left",
        fontsize=10.5,
        color="#5b6675",
        pad=10,
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#dfe3ea",
        fontsize=9.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "evolution-climbs-to-sota.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out.name}  (sampling plateau={sampling_plateau:.4f})")


if __name__ == "__main__":
    main()
