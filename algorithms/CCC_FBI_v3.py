"""
3C-FBI v3: Hybrid — full sparse vote map + localized search around top-N peaks.

Design (between v1 and v2):
  - v1 keeps only the top-5 peak weights (info loss; sparse 5x5x5 conv often
    degenerates to argmax of the top-5).
  - v2 keeps every vote in a dense volume, but sweeps uniform_filter over the
    entire volume (~300 ms at Exp B2 q=0 scale).
  - v3 keeps every vote in a sparse Counter, then only evaluates the k x k x k
    cube around each top-N peak. Cost: N * k^3 lookups per call.

Triplet-filtering layer is reused unchanged from v1's vectorized_XYR
(same three CIBICA filters: D == 0, symmetric +-20 px, radius range).

Public API matches v1's `ccc_fbi` exactly.
"""

import numpy as np
import random
from collections import Counter
from itertools import combinations

from algorithms.CCC_FBI import vectorized_XYR


def ccc_fbi_v3(edgels, Nmax=5000, xmax=50, ymax=50,
               rmin=4, rmax=30, minval=-20,
               top_n=5, cube_size=5):
    """Hybrid 3C-FBI: full vote map + localized k-cube search around top-N peaks.

    Parameters
    ----------
    edgels : numpy.ndarray, shape (n, 2)
        Edge points (x, y).
    Nmax : int
        Maximum number of random triplets (default: 5000).
    xmax, ymax : float
        Image bounds (default: 50).
    rmin, rmax : float
        Allowed radius range (default: 4, 30 — CIBICA values).
    minval : float
        Lower bound on center coordinates (default: -20 — symmetric +-20 tol.).
    top_n : int
        Number of most-voted peaks to consider (default: 5).
    cube_size : int
        Side of the cube around each peak; total cells per peak = cube_size**3
        (default: 5 → 125 cells per peak; matches v1's effective ±2 reach
        and gives the best accuracy across noise levels in MC tests).

    Returns
    -------
    center : numpy.ndarray, shape (2,)
        [xc, yc] or [-1, -1] on failure.
    radius : float
        rc or -1 on failure.
    """
    if isinstance(edgels, bool) or len(edgels) < 3:
        return np.array([-1, -1]), -1

    # ---- Step 1/2a: triplet sampling --------------------------------------
    combi = list(combinations(np.arange(len(edgels)), 3))
    N = min(Nmax, len(combi))
    sample = np.array(random.sample(combi, N))
    p1 = edgels[sample[:, 0]]
    p2 = edgels[sample[:, 1]]
    p3 = edgels[sample[:, 2]]

    # ---- Step 2b: circle fitting + CIBICA filters (reused from v1) --------
    cx, cy, r = vectorized_XYR(p1, p2, p3,
                               xmax=xmax, ymax=ymax,
                               rmin=rmin, rmax=rmax, minval=minval)
    # vectorized_XYR returns np.array([-1, -1]) sentinels on total failure
    if len(cx) == 0 or cx[0] == -1:
        return np.array([-1, -1]), -1

    # ---- Step 2c: full sparse vote map via Counter ------------------------
    points = np.column_stack((np.round(cx).astype(np.int32),
                              np.round(cy).astype(np.int32),
                              np.round(r).astype(np.int32)))
    V = Counter(map(tuple, points))

    # ---- Step 3: top-N peaks ----------------------------------------------
    top_peaks = V.most_common(top_n)
    if not top_peaks:
        return np.array([-1, -1]), -1

    # ---- Step 4: score + centroid over k**3 cube around each peak ---------
    half = cube_size // 2
    offsets = [(dx, dy, dr)
               for dx in range(-half, half + 1)
               for dy in range(-half, half + 1)
               for dr in range(-half, half + 1)]

    best_score = -1.0
    best_centroid = None
    for peak_coord, _ in top_peaks:
        score = 0.0
        wx = wy = wr = 0.0
        px, py, pr = peak_coord
        for dx, dy, dr in offsets:
            cell = (px + dx, py + dy, pr + dr)
            w = V.get(cell, 0)
            if w > 0:
                score += w
                wx += cell[0] * w
                wy += cell[1] * w
                wr += cell[2] * w
        if score > best_score:
            best_score = score
            best_centroid = (wx / score, wy / score, wr / score)

    # ---- Step 5: return ----------------------------------------------------
    if best_centroid is None:
        return np.array([-1, -1]), -1
    xc, yc, rc = best_centroid
    return np.array([float(xc), float(yc)]), float(rc)


# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    np.random.seed(0)
    random.seed(0)

    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        print("Testing 3C-FBI v3 (hybrid: full vote map + localized cube search)")
        print("=" * 60)

        # Test 1 — perfect circle
        print("\nTest 1: perfect circle")
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        true_c, true_r = (25.0, 25.0), 10.0
        edgels = np.column_stack([true_c[0] + true_r * np.cos(theta),
                                  true_c[1] + true_r * np.sin(theta)])
        c, r = ccc_fbi_v3(edgels, xmax=50, ymax=50, rmin=4, rmax=30)
        print(f"  True:     center={true_c}, r={true_r}")
        print(f"  Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 2 — noisy circle
        print("\nTest 2: noisy circle (sigma=0.5)")
        rng = np.random.default_rng(1)
        edgels_noisy = edgels + rng.normal(0, 0.5, edgels.shape)
        c, r = ccc_fbi_v3(edgels_noisy, xmax=50, ymax=50, rmin=4, rmax=30)
        print(f"  Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 3 — partial arc with outliers, default cube_size=3 then 5
        print("\nTest 3: partial arc (90 deg) with 20 outliers")
        theta_p = np.linspace(0, np.pi / 2, 50)
        arc = np.column_stack([true_c[0] + true_r * np.cos(theta_p),
                               true_c[1] + true_r * np.sin(theta_p)])
        outl = rng.uniform(0, 50, (20, 2))
        edgels_p = np.vstack([arc, outl])
        c, r = ccc_fbi_v3(edgels_p, Nmax=3000, xmax=50, ymax=50,
                          rmin=4, rmax=30, cube_size=3)
        print(f"  cube=3:  center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")
        c, r = ccc_fbi_v3(edgels_p, Nmax=3000, xmax=50, ymax=50,
                          rmin=4, rmax=30, cube_size=5)
        print(f"  cube=5:  center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 4 — too few points
        print("\nTest 4: too few points (should fail safely)")
        edgels_few = np.array([[10.0, 10.0], [20.0, 20.0]])
        c, r = ccc_fbi_v3(edgels_few)
        print(f"  Result: center={tuple(c)}, r={r}  (expect (-1,-1), -1)")

        print("\n" + "=" * 60)
        print("Testing complete.")
    else:
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        c, r = ccc_fbi_v3(edgels)
        print(f"Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")
