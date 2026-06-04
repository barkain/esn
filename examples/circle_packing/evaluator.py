# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Circle-packing validation helper for the ESN example.

Provides ``_validate_packing`` and the ``N_CIRCLES`` constant used by
``domain.py`` to score a candidate packing: 26 circles in a unit square,
goal maximize the sum of radii.

Geometry checks ported from SakanaAI/ShinkaEvolve circle_packing example.
"""

from typing import Optional

import numpy as np

N_CIRCLES = 26
OVERLAP_EPS = 1e-6


def _validate_packing(
    centers: np.ndarray,
    radii: np.ndarray,
) -> Optional[str]:
    """Return an error string if the packing is invalid, else None."""
    # Shape checks
    if centers.shape != (N_CIRCLES, 2):
        return f"centers shape {centers.shape}, expected ({N_CIRCLES}, 2)"
    if radii.shape != (N_CIRCLES,):
        return f"radii shape {radii.shape}, expected ({N_CIRCLES},)"

    # Finiteness
    if not np.all(np.isfinite(centers)):
        return "centers contain non-finite values"
    if not np.all(np.isfinite(radii)):
        return "radii contain non-finite values"

    # Non-negative radii
    if np.any(radii < 0):
        return "negative radii detected"

    # Boundary: each circle must be inside [0, 1]^2
    for i in range(N_CIRCLES):
        x, y = centers[i]
        r = radii[i]
        if x - r < -OVERLAP_EPS or y - r < -OVERLAP_EPS:
            return f"circle {i} exceeds lower boundary"
        if x + r > 1 + OVERLAP_EPS or y + r > 1 + OVERLAP_EPS:
            return f"circle {i} exceeds upper boundary"

    # No overlaps: dist(c_i, c_j) >= r_i + r_j - eps
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < radii[i] + radii[j] - OVERLAP_EPS:
                return (
                    f"circles {i} and {j} overlap: dist={dist:.6f}, "
                    f"r_i+r_j={radii[i] + radii[j]:.6f}"
                )

    return None
