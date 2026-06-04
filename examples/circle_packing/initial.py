# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Initial (seed) solution for the circle packing benchmark.

A simple ring-based layout that places 26 circles in a unit square:
- 1 circle at the center
- 8 circles in an inner ring
- 17 circles in an outer ring
Radii are computed greedily to avoid overlap and stay within bounds.

This scores modestly and serves as the starting point for ESN-guided evolution.
"""

INITIAL_SOLUTION = '''
import numpy as np

def construct_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.

    Simple ring-based layout:
    - 1 central circle
    - 8 inner ring
    - 17 outer ring

    Returns:
        (centers, radii, sum_of_radii)
    """
    n = 26
    centers = np.zeros((n, 2))

    # Central circle
    centers[0] = [0.5, 0.5]

    # Inner ring: 8 circles at radius 0.25 from center
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[1 + i] = [0.5 + 0.25 * np.cos(angle),
                          0.5 + 0.25 * np.sin(angle)]

    # Outer ring: 17 circles at radius 0.42 from center
    for i in range(17):
        angle = 2 * np.pi * i / 17 + np.pi / 17  # offset to stagger
        centers[9 + i] = [0.5 + 0.42 * np.cos(angle),
                          0.5 + 0.42 * np.sin(angle)]

    # Clip to keep centers safely inside [0, 1]
    centers = np.clip(centers, 0.02, 0.98)

    # Greedy radius computation
    radii = np.zeros(n)

    # Start with max possible radius (distance to nearest wall)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively shrink to avoid overlaps
    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    # Shrink both proportionally
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale

    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
'''


def generate_initial_packing(n_circles: int = 26) -> list[tuple[float, float, float]]:
    """Execute the ring-based layout and return list of (x, y, r) tuples.

    This is a callable version of INITIAL_SOLUTION for programmatic use
    (e.g. by the v2 domain factory).
    """
    import numpy as np

    centers = np.zeros((n_circles, 2))

    # Central circle
    centers[0] = [0.5, 0.5]

    # Inner ring: 8 circles at radius 0.25 from center
    n_inner = min(8, n_circles - 1)
    for i in range(n_inner):
        angle = 2 * np.pi * i / 8
        centers[1 + i] = [0.5 + 0.25 * np.cos(angle), 0.5 + 0.25 * np.sin(angle)]

    # Outer ring: remaining circles at radius 0.42 from center
    n_outer = n_circles - 1 - n_inner
    for i in range(n_outer):
        angle = 2 * np.pi * i / max(n_outer, 1) + np.pi / max(n_outer, 1)
        centers[1 + n_inner + i] = [0.5 + 0.42 * np.cos(angle), 0.5 + 0.42 * np.sin(angle)]

    # Clip to keep centers safely inside [0, 1]
    centers = np.clip(centers, 0.02, 0.98)

    # Greedy radius computation
    radii = np.zeros(n_circles)
    for i in range(n_circles):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Iteratively shrink to avoid overlaps
    for _ in range(5):
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale

    radii = np.maximum(radii, 0.0)

    return [(float(centers[i, 0]), float(centers[i, 1]), float(radii[i])) for i in range(n_circles)]
