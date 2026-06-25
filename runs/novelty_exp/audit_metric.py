"""Audit B: is a ~2.5 circle-packing score real or gamed?

Feeds the SAME evaluator the bias-free domain uses several constructed
packings and prints (score, success, violations) plus an independent
geometric re-check, to confirm the metric rejects gaming and only awards
high scores to genuinely valid packings.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
from circle_packing.domain import evaluate_circle_packing_artifact  # noqa: E402
from circle_packing.evaluator import _validate_packing, N_CIRCLES  # noqa: E402


def independent_audit(centers, radii):
    """Recompute validity from scratch, return (ok, sum_radii, reason)."""
    centers = np.asarray(centers, float)
    radii = np.asarray(radii, float)
    if centers.shape != (N_CIRCLES, 2) or radii.shape != (N_CIRCLES,):
        return False, float(np.sum(radii)), "shape"
    if np.any(radii < -1e-12) or not np.all(np.isfinite(radii)):
        return False, float(np.sum(radii)), "bad radii"
    for i in range(N_CIRCLES):
        x, y = centers[i]; r = radii[i]
        if x - r < -1e-6 or y - r < -1e-6 or x + r > 1 + 1e-6 or y + r > 1 + 1e-6:
            return False, float(np.sum(radii)), f"circle {i} out of bounds"
    worst = 0.0
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d = float(np.hypot(*(centers[i] - centers[j])))
            overlap = radii[i] + radii[j] - d
            worst = max(worst, overlap)
            if overlap > 1e-6:
                return False, float(np.sum(radii)), f"overlap {i},{j}={overlap:.4g}"
    return True, float(np.sum(radii)), f"valid (worst_overlap={worst:.2e})"


def grid_5x5_r01():
    """5x5 grid of r=0.1 circles (touching) + 1 degenerate circle -> sum=2.5."""
    cs, rs = [], []
    for gx in (0.1, 0.3, 0.5, 0.7, 0.9):
        for gy in (0.1, 0.3, 0.5, 0.7, 0.9):
            cs.append((gx, gy)); rs.append(0.1)
    cs.append((0.2, 0.2)); rs.append(0.0)  # 26th, r=0 in interstitial gap (dist~0.141>0.1)
    return np.array(cs), np.array(rs)


def gamed_overlap():
    """26 fat circles all stacked -> sum huge but massively overlapping."""
    cs = np.full((26, 2), 0.5)
    rs = np.full(26, 0.4)  # sum=10.4 if it were not rejected
    return cs, rs


def gamed_oob():
    """26 circles partly outside the square -> should be rejected."""
    cs = np.full((26, 2), 0.0)
    rs = np.full(26, 0.3)
    return cs, rs


for name, (c, r) in [
    ("grid_5x5_r0.1 (expect ~2.5 VALID)", grid_5x5_r01()),
    ("gamed_overlap (expect score 0)", gamed_overlap()),
    ("gamed_out_of_bounds (expect score 0)", gamed_oob()),
]:
    res = evaluate_circle_packing_artifact((c, r))
    ok, s, reason = independent_audit(c, r)
    val_err = _validate_packing(np.asarray(c, float), np.asarray(r, float))
    print(f"[{name}]")
    print(f"  evaluator: score={res.score:.6f} success={res.success} "
          f"violations={list(res.diagnostics.violations)}")
    print(f"  validator: {val_err or 'None (valid)'}")
    print(f"  independent audit: ok={ok} raw_sum={s:.6f} -> {reason}")
    print()
