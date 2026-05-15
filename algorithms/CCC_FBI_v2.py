"""
3C-FBI v2: Dense vote-map circle fitting with multi-k windowed peak detection.

Differences from v1 (algorithms/CCC_FBI.py):
  - Keeps ALL surviving triplet votes in a dense accumulator V
    (v1 keeps only the top-5 most frequent integer cells).
  - Replaces the sparse 3x3x3-around-peaks + 5x5x5 convolution by a
    sweep of uniform k x k x k windows over V for k in k_values.
  - Picks the window position with the largest summed weight, then
    returns the value-weighted centroid of the votes inside that window.
  - For each k the normalised density S/k^3 is compared and the k
    with the highest density wins (ties broken by smaller k).

Triplet-filtering layer is reused unchanged from v1's vectorized_XYR
(same three CIBICA filters: D == 0, symmetric +-20 px, radius range).

Public API matches v1's `ccc_fbi` exactly.
"""

import numpy as np
import random
from itertools import combinations
from scipy.ndimage import uniform_filter

from algorithms.CCC_FBI import vectorized_XYR


def _windowed_centroid(V, k):
    """Find the k-cube window of V with the largest summed mass and
    return (S, mu) — the sum and value-weighted centroid of that window.

    Uses scipy.ndimage.uniform_filter (returns the MEAN over the window),
    multiplied by k**3 to recover the SUM.
    """
    Vf = V.astype(np.float64)

    # Window sum at every cell (cell = window center)
    S = uniform_filter(Vf, size=k, mode='constant') * (k ** 3)

    # Weighted moments per axis
    X, Y, R = np.indices(V.shape, dtype=np.float64)
    M_x = uniform_filter(Vf * X, size=k, mode='constant') * (k ** 3)
    M_y = uniform_filter(Vf * Y, size=k, mode='constant') * (k ** 3)
    M_r = uniform_filter(Vf * R, size=k, mode='constant') * (k ** 3)

    flat_idx = int(np.argmax(S))
    c = np.unravel_index(flat_idx, S.shape)
    s_star = float(S[c])
    if s_star <= 0:
        return 0.0, None

    mu = np.array([M_x[c], M_y[c], M_r[c]]) / s_star
    return s_star, mu


def ccc_fbi_v2(edgels, Nmax=5000, xmax=50, ymax=50,
               rmin=4, rmax=30, minval=-20,
               k_values=(2, 3, 4, 5)):
    """Dense vote-map circle fitting with multi-k windowed peak detection.

    Parameters
    ----------
    edgels : numpy.ndarray, shape (n, 2)
        Edge points (x, y).
    Nmax : int
        Maximum number of random triplets to sample (default: 5000).
    xmax, ymax : float
        Image bounds (default: 50).
    rmin, rmax : float
        Allowed radius range (default: 4, 30 — CIBICA values).
    minval : float
        Lower bound on center coordinates (default: -20 — symmetric +-20 tol.).
    k_values : tuple of int
        Window sizes to try; the best k is chosen by max normalised density
        S / k**3 (default: (2, 3, 4, 5)).

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

    # ---- Step 2c: round and accumulate into dense V -----------------------
    pts = np.column_stack((np.round(cx).astype(np.int32),
                           np.round(cy).astype(np.int32),
                           np.round(r).astype(np.int32)))

    kmax = max(k_values)
    mn = pts.min(axis=0) - kmax
    mx = pts.max(axis=0) + kmax
    shape = tuple((mx - mn + 1).astype(int))

    V = np.zeros(shape, dtype=np.int32)
    shifted = pts - mn
    np.add.at(V, (shifted[:, 0], shifted[:, 1], shifted[:, 2]), 1)

    # ---- Step 3: multi-k windowed peak + centroid -------------------------
    best_density = -1.0
    best_k = None
    best_mu_shifted = None

    for k in k_values:
        s_star, mu = _windowed_centroid(V, k)
        if mu is None:
            continue
        density = s_star / (k ** 3)
        # Tie-break by smaller k (tighter window preferred)
        if (density > best_density) or (
                density == best_density and (best_k is None or k < best_k)):
            best_density = density
            best_k = k
            best_mu_shifted = mu

    if best_mu_shifted is None:
        return np.array([-1, -1]), -1

    # ---- Step 5: un-shift and return --------------------------------------
    mu = best_mu_shifted + mn
    xc, yc, rc = float(mu[0]), float(mu[1]), float(mu[2])
    return np.array([xc, yc]), rc


# ---------------------------------------------------------------------------
# Built-in tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    np.random.seed(0)
    random.seed(0)

    if len(sys.argv) < 2 or sys.argv[1] == 'test':
        print("Testing 3C-FBI v2 (dense + multi-k windowed peak)")
        print("=" * 60)

        # Test 1 — perfect circle
        print("\nTest 1: perfect circle")
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        true_c, true_r = (25.0, 25.0), 10.0
        edgels = np.column_stack([true_c[0] + true_r * np.cos(theta),
                                  true_c[1] + true_r * np.sin(theta)])
        c, r = ccc_fbi_v2(edgels, xmax=50, ymax=50, rmin=4, rmax=30)
        print(f"  True:     center={true_c}, r={true_r}")
        print(f"  Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 2 — noisy circle
        print("\nTest 2: noisy circle (sigma=0.5)")
        rng = np.random.default_rng(1)
        edgels_noisy = edgels + rng.normal(0, 0.5, edgels.shape)
        c, r = ccc_fbi_v2(edgels_noisy, xmax=50, ymax=50, rmin=4, rmax=30)
        print(f"  Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 3 — partial arc with outliers
        print("\nTest 3: partial arc (90 deg) with 20 outliers")
        theta_p = np.linspace(0, np.pi / 2, 50)
        arc = np.column_stack([true_c[0] + true_r * np.cos(theta_p),
                               true_c[1] + true_r * np.sin(theta_p)])
        outl = rng.uniform(0, 50, (20, 2))
        edgels_p = np.vstack([arc, outl])
        c, r = ccc_fbi_v2(edgels_p, Nmax=3000, xmax=50, ymax=50,
                          rmin=4, rmax=30)
        print(f"  Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")

        # Test 4 — too few points
        print("\nTest 4: too few points (should fail safely)")
        edgels_few = np.array([[10.0, 10.0], [20.0, 20.0]])
        c, r = ccc_fbi_v2(edgels_few)
        print(f"  Result: center={tuple(c)}, r={r}  (expect (-1,-1), -1)")

        print("\n" + "=" * 60)
        print("Testing complete.")
    else:
        edgels = np.loadtxt(sys.argv[1], delimiter=',')
        c, r = ccc_fbi_v2(edgels)
        print(f"Detected: center=({c[0]:.2f}, {c[1]:.2f}), r={r:.2f}")
