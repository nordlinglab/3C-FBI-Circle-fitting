"""
compare_implementations.py
--------------------------
Compare 5 implementations of the 3C-FBI circle-fitting algorithm.

Files compared
--------------
  algorithms/CCC_FBI.py       v1 — original (kept for ablation)
  algorithms/CCC_FBI_v2.py    v2 — dense vote map + multi-k windowed
  algorithms/CCC_FBI_v3.py    v3 — PUBLISHED variant (cube_size=3)
  algorithms/CCC_FBI_ChatGPT.py
  algorithms/CCC_FBI_Gemini.py

v1/v2/v3 use numba (not installed). The script mocks it so @jit becomes
a no-op — results are identical, speed slightly lower for v1.

Usage
-----
    conda activate poseestimation
    python compare_implementations.py
"""

import sys
import os
import io
import time
import contextlib
from unittest.mock import MagicMock

# Mock numba before any algorithm imports
_numba_mock = MagicMock()
_numba_mock.jit = lambda *args, **kwargs: lambda f: f
sys.modules['numba'] = _numba_mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── imports ────────────────────────────────────────────────────────────────

AVAILABLE = {}

try:
    from algorithms.CCC_FBI import ccc_fbi as _v1
    AVAILABLE['v1 (original)'] = 'v1'
except Exception as e:
    print(f"  [SKIP] v1: {e}")

try:
    from algorithms.CCC_FBI_v2 import ccc_fbi_v2 as _v2
    AVAILABLE['v2 (dense)'] = 'v2'
except Exception as e:
    print(f"  [SKIP] v2: {e}")

try:
    from algorithms.CCC_FBI_v3 import ccc_fbi_v3 as _v3
    AVAILABLE['v3 ★ published'] = 'v3'
except Exception as e:
    print(f"  [SKIP] v3: {e}")

try:
    from algorithms.CCC_FBI_ChatGPT import fit_circle_3cfbi as _chatgpt
    AVAILABLE['ChatGPT'] = 'chatgpt'
except Exception as e:
    print(f"  [SKIP] ChatGPT: {e}")

try:
    from algorithms.CCC_FBI_Gemini import c3_fbi as _gemini
    AVAILABLE['Gemini'] = 'gemini'
except Exception as e:
    print(f"  [SKIP] Gemini: {e}")

# ── shared parameters ──────────────────────────────────────────────────────

X0, Y0, R0   = 50.0, 50.0, 20.0   # ground truth circle
IMG_W, IMG_H = 121, 121            # image size (x_max=120, y_max=120)
RMIN, RMAX   = 4, 40
NMAX         = 5000
TOP_N        = 5
TAU          = 1                   # cube_size = 2*tau+1 = 3  (published)
N_MC         = 50                  # Monte Carlo trials per scenario

# ── helpers ────────────────────────────────────────────────────────────────

def make_edgels(x0, y0, r0, n_pts, noise_std=0.0,
                n_outliers=0, arc_fraction=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi * arc_fraction, n_pts, endpoint=False)
    pts = np.column_stack([x0 + r0 * np.cos(theta),
                           y0 + r0 * np.sin(theta)])
    if noise_std > 0:
        pts += rng.normal(0, noise_std, pts.shape)
    if n_outliers > 0:
        pts = np.vstack([pts, rng.uniform([0, 0], [IMG_W, IMG_H], (n_outliers, 2))])
    return pts


def edgels_to_binary(edgels):
    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for x, y in edgels:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < IMG_H and 0 <= xi < IMG_W:
            img[yi, xi] = 255
    return img


def center_err(xc, yc):
    return float(np.hypot(xc - X0, yc - Y0))

def radius_err(rc):
    return float(abs(rc - R0))


# ── per-implementation wrappers ────────────────────────────────────────────

def _call_v1(edgels):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # suppress v1's print()
        c, r = _v1(edgels, Nmax=NMAX, xmax=IMG_W - 1, ymax=IMG_H - 1,
                   rmin=RMIN, rmax=RMAX, top_n=TOP_N)
    if r == -1:
        return None
    return c[0], c[1], r

def _call_v2(edgels):
    c, r = _v2(edgels, Nmax=NMAX, xmax=IMG_W - 1, ymax=IMG_H - 1,
               rmin=RMIN, rmax=RMAX)
    if r == -1:
        return None
    return c[0], c[1], r

def _call_v3(edgels):
    c, r = _v3(edgels, Nmax=NMAX, xmax=IMG_W - 1, ymax=IMG_H - 1,
               rmin=RMIN, rmax=RMAX, top_n=TOP_N, cube_size=2 * TAU + 1)
    if r == -1:
        return None
    return c[0], c[1], r

def _call_chatgpt(edgels):
    img = edgels_to_binary(edgels)
    try:
        est = _chatgpt(img, x_max=IMG_W - 1, y_max=IMG_H - 1,
                       r_min=RMIN, r_max=RMAX,
                       n_tri=NMAX, n_peaks=TOP_N, tau=TAU)
        return est.x, est.y, est.r
    except Exception:
        return None

def _call_gemini(edgels):
    img = edgels_to_binary(edgels)
    est = _gemini(img, r_range=(RMIN, RMAX), n_tri=NMAX, n_peaks=TOP_N, tau=TAU)
    if est is None:
        return None
    return float(est[0]), float(est[1]), float(est[2])


CALLERS = {
    'v1': _call_v1, 'v2': _call_v2, 'v3': _call_v3,
    'chatgpt': _call_chatgpt, 'gemini': _call_gemini,
}

# ── scenarios ──────────────────────────────────────────────────────────────

SCENARIOS = [
    ("Full circle, no noise (n=100)",
     dict(n_pts=100, noise_std=0.0, n_outliers=0,  arc_fraction=1.0)),
    ("Full circle, σ=1 px (n=100)",
     dict(n_pts=100, noise_std=1.0, n_outliers=0,  arc_fraction=1.0)),
    ("Full circle, σ=1 + 20 outliers (n=100)",
     dict(n_pts=100, noise_std=1.0, n_outliers=20, arc_fraction=1.0)),
    ("Semicircle, σ=1 px (n=50)",
     dict(n_pts=50,  noise_std=1.0, n_outliers=0,  arc_fraction=0.5)),
]

# ── run ────────────────────────────────────────────────────────────────────

def run_scenario(scenario_name, kw):
    W = 72
    print(f"\n{'─' * W}")
    print(f"  {scenario_name}")
    print(f"{'─' * W}")
    hdr = f"{'Method':<22} {'Ctr err (px)':>13} {'Rad err (px)':>13} {'Time (ms)':>10} {'Fail %':>7}"
    print(hdr)
    print('─' * len(hdr))

    for label, key in AVAILABLE.items():
        caller = CALLERS[key]
        c_errs, r_errs, times, fails = [], [], [], 0
        for seed in range(N_MC):
            rng = np.random.default_rng(seed)
            edgels = make_edgels(X0, Y0, R0, rng=rng, **kw)
            t0 = time.perf_counter()
            try:
                result = caller(edgels)
            except Exception:
                result = None
            dt = (time.perf_counter() - t0) * 1000
            if result is None:
                fails += 1
            else:
                xc, yc, rc = result
                c_errs.append(center_err(xc, yc))
                r_errs.append(radius_err(rc))
                times.append(dt)

        n_ok = N_MC - fails
        if n_ok > 0:
            ce = np.mean(c_errs)
            re = np.mean(r_errs)
            tm = np.mean(times)
            print(f"{label:<22} {ce:>13.4f} {re:>13.4f} {tm:>10.2f} {100*fails/N_MC:>6.1f}%")
        else:
            print(f"{label:<22} {'—':>13} {'—':>13} {'—':>10} {100.0:>6.1f}%")


def main():
    W = 72
    print('=' * W)
    print("  3C-FBI Implementation Comparison")
    print(f"  Ground truth: center=({X0},{Y0}), r={R0}  |  {N_MC} MC trials")
    print(f"  Params: Nmax={NMAX}, top_n={TOP_N}, cube_size={2*TAU+1}, "
          f"r=[{RMIN},{RMAX}]")
    print('=' * W)

    for name, kw in SCENARIOS:
        run_scenario(name, kw)

    print(f"\n{'=' * W}")
    print("  ★ v3 with cube_size=3 is the published variant (CLAUDE.md)")
    print("  v1 note: @jit replaced by no-op mock (numba not installed)")
    print('=' * W)


if __name__ == '__main__':
    main()
