"""
main_3C_FBI_V2.py — Focused v1 vs v3-cube3 vs v3-cube5 comparison for 3C-FBI.

Compares:
  - '3C-FBI'       : algorithms.CCC_FBI.ccc_fbi             (v1, top-5 + sparse 5x5x5 conv)
  - '3C-FBI-v3-c3' : algorithms.CCC_FBI_v3.ccc_fbi_v3 (cube_size=3, ±1 reach)
  - '3C-FBI-v3-c5' : algorithms.CCC_FBI_v3.ccc_fbi_v3 (cube_size=5, ±2 reach)

Three experiments (same data and parameters as main_3C_FBI.py):
  A  : 144 frames x 18 preprocessing configs (real-world Parkinson's data)
  B1 : Synthetic semicircle, varying outliers
  B2 : Synthetic full circle, noise x outliers x quantization grid

Output folder: CCC_FBI_v1_vs_v2_results/

Usage:
    cd /Users/erc/Documents/3C-FBI-Circle-fitting
    /opt/homebrew/Caskroom/miniforge/base/envs/poseestimation/bin/python main_3C_FBI_V2.py
    /opt/homebrew/Caskroom/miniforge/base/envs/poseestimation/bin/python main_3C_FBI_V2.py fast
"""

import contextlib
import io
import math as m
import os
import sys
import time
from datetime import date

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from algorithms.CCC_FBI    import ccc_fbi
from algorithms.CCC_FBI_v3 import ccc_fbi_v3
from algorithms.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)


# ============================================================================
# Constants and plot styling
# ============================================================================

DATE       = date.today().strftime('%Y%m%d')
OUTPUT_DIR = 'CCC_FBI_v1_vs_v2_results'
FAST       = len(sys.argv) > 1 and sys.argv[1] == 'fast'

METHODS = ['3C-FBI', '3C-FBI-v3-c3', '3C-FBI-v3-c5']

COLORS = {
    '3C-FBI':       '#1f77b4',   # blue    (v1, reference)
    '3C-FBI-v3-c3': '#ff7f0e',   # orange  (v3 cube_size=3, ±1 reach)
    '3C-FBI-v3-c5': '#2ca02c',   # green   (v3 cube_size=5, ±2 reach)
}
MARKERS = {
    '3C-FBI':       'o',
    '3C-FBI-v3-c3': 's',
    '3C-FBI-v3-c5': '^',
}
LINESTYLES = {
    '3C-FBI':       'solid',
    '3C-FBI-v3-c3': 'dashed',
    '3C-FBI-v3-c5': 'dashdot',
}

# Synthetic experiment parameters (mirror main_3C_FBI.py)
N_ITER_B            = 100 if not FAST else 10
B1_X0, B1_Y0, B1_R0 = 50,  60,  100
B2_X0, B2_Y0, B2_R0 = 120, 120, 120
B2_N_POINTS         = 100

BEST_GL = ['GL80', 'GL82', 'GL84']

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'lines.markersize':  6,
})


# ============================================================================
# Jaccard index (analytical exact formula) — copied from main_3C_FBI.py
# ============================================================================

def jaccard_circles(x1, y1, r1, x2, y2, r2):
    if r1 <= 0 or r2 <= 0 or any(np.isnan([x1, y1, r1, x2, y2, r2])):
        return 0.0
    d = m.hypot(x2 - x1, y2 - y1)
    if d >= r1 + r2:
        return 0.0
    if d + min(r1, r2) <= max(r1, r2):
        a_min, a_max = m.pi * min(r1, r2)**2, m.pi * max(r1, r2)**2
        return a_min / a_max
    R, r = max(r1, r2), min(r1, r2)
    cos1 = (d*d + R*R - r*r) / (2*d*R)
    cos2 = (d*d + r*r - R*R) / (2*d*r)
    cos1 = max(-1.0, min(1.0, cos1))
    cos2 = max(-1.0, min(1.0, cos2))
    inter = R*R*m.acos(cos1) - 0.5 * m.sqrt(max(0,
              (-d+R+r) * (d+R-r) * (d-R+r) * (d+R+r)))
    inter += r*r*m.acos(cos2)
    union = m.pi*r1*r1 + m.pi*r2*r2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# Method dispatch
# ============================================================================

def _call_v1(edgels, xmax, ymax, rmax=40, rmin=4):
    with contextlib.redirect_stdout(io.StringIO()):
        center, r = ccc_fbi(edgels, Nmax=5000, xmax=xmax, ymax=ymax,
                            rmin=rmin, rmax=rmax)
    return np.array(center, dtype=float), float(r)


def _call_v3(edgels, xmax, ymax, rmax=40, rmin=4, cube_size=5):
    center, r = ccc_fbi_v3(edgels, Nmax=5000, xmax=xmax, ymax=ymax,
                           rmin=rmin, rmax=rmax, cube_size=cube_size)
    return np.array(center, dtype=float), float(r)


def run_method_A(method, edgels, xmax, ymax):
    """Real-image edgels are [row, col]; v1/v3 return [col, row]."""
    t0 = time.perf_counter()
    try:
        if method == '3C-FBI':
            center, r = _call_v1(edgels, xmax, ymax)
        elif method == '3C-FBI-v3-c3':
            center, r = _call_v3(edgels, xmax, ymax, cube_size=3)
        elif method == '3C-FBI-v3-c5':
            center, r = _call_v3(edgels, xmax, ymax, cube_size=5)
        else:
            return np.array([-1., -1.]), -1., 0.
    except Exception:
        return np.array([-1., -1.]), -1., time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    center  = np.array(center, dtype=float)
    r       = float(r)
    if r <= 0 or np.any(np.isnan(center)):
        return np.array([-1., -1.]), -1., elapsed
    return center, r, elapsed


def run_method_B(method, points, xmax, ymax, rmax=300, rmin=4):
    t0 = time.perf_counter()
    try:
        if method == '3C-FBI':
            center, r = _call_v1(points, xmax, ymax, rmax=rmax, rmin=rmin)
        elif method == '3C-FBI-v3-c3':
            center, r = _call_v3(points, xmax, ymax, rmax=rmax, rmin=rmin, cube_size=3)
        elif method == '3C-FBI-v3-c5':
            center, r = _call_v3(points, xmax, ymax, rmax=rmax, rmin=rmin, cube_size=5)
        else:
            return np.nan, np.nan, -1., 0.
    except Exception:
        return np.nan, np.nan, -1., time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    cx, cy  = float(center[0]), float(center[1])
    r       = float(r)
    if r <= 0 or np.isnan(cx) or np.isnan(cy):
        return np.nan, np.nan, -1., elapsed
    return cx, cy, r, elapsed


# ============================================================================
# Synthetic generators — copied verbatim from main_3C_FBI.py
# ============================================================================

def generate_semicircle_points(x0=50, y0=60, r0=100, n_points=50,
                               noise_std=1.0, n_outliers=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    theta  = rng.uniform(0, np.pi, n_points)
    noise1 = rng.normal(0, noise_std, n_points)
    x = x0 + (r0 + noise1) * np.cos(theta)
    y = y0 + (r0 + noise1) * np.sin(theta)
    if n_outliers > 0:
        n_outliers = min(n_outliers, n_points)
        for idx in rng.choice(n_points, n_outliers, replace=False):
            sign   = 2 * round(rng.uniform(0, 1)) - 1
            factor = rng.uniform(9, 10) * noise_std * sign
            x[idx] = x0 + (r0 + factor) * np.cos(theta[idx])
            y[idx] = y0 + (r0 + factor) * np.sin(theta[idx])
    return np.column_stack([x, y])


def generate_circle_points(x0=120, y0=120, r0=120, n_points=100,
                           noise_std=0.0, n_outliers=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    theta   = rng.uniform(0, 2 * np.pi, n_points)
    r_noise = rng.normal(0, noise_std, n_points)
    x = x0 + (r0 + r_noise) * np.cos(theta)
    y = y0 + (r0 + r_noise) * np.sin(theta)
    if n_outliers > 0:
        n_outliers = min(n_outliers, n_points)
        for idx in rng.choice(n_points, n_outliers, replace=False):
            sign   = 2 * round(rng.uniform(0, 1)) - 1
            factor = rng.uniform(5, 20) * noise_std * sign
            x[idx] = x0 + (r0 + factor) * np.cos(theta[idx])
            y[idx] = y0 + (r0 + factor) * np.sin(theta[idx])
    return np.column_stack([x, y])


def apply_quantization(points, q):
    if q == 0:
        return points
    return np.floor(points / q).astype(int) * q


# ============================================================================
# Experiment A — Real-world clinical data
# ============================================================================

def run_experiment_A():
    gt        = pd.read_csv('data/Ground_Truth.csv')
    files     = gt['Filename'].tolist()
    XGT_arr   = gt['X'].to_numpy()
    YGT_arr   = gt['Y'].to_numpy()
    RGT_arr   = gt['R'].to_numpy()
    configs   = get_preprocessing_configs()
    cfg_names = [c['name'] for c in configs]

    n_img, n_cfg, n_meth = len(files), len(configs), len(METHODS)
    Jaccard = np.zeros((n_meth, n_cfg, n_img))
    AD      = np.zeros_like(Jaccard)
    RMSE    = np.zeros_like(Jaccard)
    Time_s  = np.zeros_like(Jaccard)

    print(f"Experiment A: {n_img} images x {n_cfg} configs x {n_meth} methods")
    print("=" * 70)
    t_start = time.time()

    for i, filename in enumerate(files):
        XGT, YGT, RGT = XGT_arr[i], YGT_arr[i], RGT_arr[i]
        BS_crop = cv2.imread(os.path.join('data', 'black_sphere_ROI', filename + '.png'))
        G_crop  = cv2.imread(os.path.join('data', 'green_back_ROI',   filename + '.png'))
        if BS_crop is None:
            print(f"  Warning: missing {filename} - skipping")
            continue

        xmax, ymax = BS_crop.shape[1], BS_crop.shape[0]

        for j, cfg in enumerate(configs):
            try:
                if cfg['green_level'] is not None:
                    _, _, edgels = preprocess_green_level(BS_crop, cfg['green_level'])
                else:
                    _, _, edgels = preprocess_median_filter(BS_crop, G_crop, cfg['median_size'])
            except Exception:
                continue
            if len(edgels) < 3:
                continue

            for k, method in enumerate(METHODS):
                center, r, elapsed = run_method_A(method, edgels, xmax, ymax)
                Time_s[k, j, i] = elapsed
                if r > 0:
                    Jaccard[k, j, i] = jaccard_circles(YGT, XGT, RGT,
                                                       center[0], center[1], r)
                    AD[k, j, i]   = np.sqrt((XGT - center[1])**2 + (YGT - center[0])**2)
                    RMSE[k, j, i] = abs(RGT - r)

        if (i + 1) % 20 == 0 or (i + 1) == n_img:
            print(f"  {i+1}/{n_img}  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    print(f"Done in {time.time()-t_start:.1f}s")
    return {'Jaccard': Jaccard, 'AD': AD, 'RMSE': RMSE, 'Time_s': Time_s,
            'config_names': cfg_names, 'filenames': files}


def save_experiment_A(res):
    cfg_names = res['config_names']
    n_meth, n_cfg, n_img = res['Jaccard'].shape

    # Raw per-method tables
    for k, method in enumerate(METHODS):
        df = pd.DataFrame(res['Jaccard'][k].T,
                          index=res['filenames'], columns=cfg_names)
        df.index.name = 'Filename'
        df.to_csv(os.path.join(OUTPUT_DIR, f'A_Jaccard_{method}_{DATE}.csv'))

    # Mean / std per (method, config) summary
    rows = []
    for k, method in enumerate(METHODS):
        for j, cfg in enumerate(cfg_names):
            J = res['Jaccard'][k, j, :]
            t_mean = res['Time_s'][k, j, :].mean()
            rows.append({
                'Method':   method,
                'Config':   cfg,
                'J_mean':   round(J.mean(), 4),
                'J_std':    round(J.std(),  4),
                'AD_mean':  round(res['AD'][k, j, :].mean(), 4),
                'RMSE_mean':round(res['RMSE'][k, j, :].mean(), 4),
                'Time_s':   round(t_mean, 6),
                'FPS':      round(1.0 / t_mean if t_mean > 0 else 0, 1),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f'A_Summary_{DATE}.csv'), index=False)

    # Best-GL table (mean over GL80/82/84)
    gl_idx = [cfg_names.index(g) for g in BEST_GL if g in cfg_names]
    best_rows = []
    for k, method in enumerate(METHODS):
        J = res['Jaccard'][k, gl_idx, :].mean(axis=0)
        t = res['Time_s'][k, gl_idx, :].mean()
        best_rows.append({
            'Method':   method,
            'J_mean':   round(J.mean(), 4),
            'J_std':    round(J.std(),  4),
            'AD_mean':  round(res['AD'][k, gl_idx, :].mean(), 4),
            'RMSE_mean':round(res['RMSE'][k, gl_idx, :].mean(), 4),
            'Time_s':   round(t, 6),
            'FPS':      round(1.0 / t if t > 0 else 0, 1),
        })
    pd.DataFrame(best_rows).to_csv(
        os.path.join(OUTPUT_DIR, f'A_BestGL_{DATE}.csv'), index=False)

    # Pairwise paired Wilcoxon over GL80/82/84 per-image scores
    j_per_method = [res['Jaccard'][k, gl_idx, :].mean(axis=0) for k in range(n_meth)]
    stat_rows = []
    for a in range(n_meth):
        for b in range(a + 1, n_meth):
            diff = j_per_method[b] - j_per_method[a]
            try:
                w_stat, p_val = wilcoxon(diff, zero_method='wilcox',
                                         alternative='two-sided')
            except ValueError:
                w_stat, p_val = float('nan'), float('nan')
            stat_rows.append({
                'Comparison':  f'{METHODS[b]} - {METHODS[a]}',
                'N':           len(diff),
                'Mean_diff':   round(diff.mean(), 4),
                'Median_diff': round(np.median(diff), 4),
                'Wilcoxon_W':  w_stat,
                'p_two_sided': p_val,
            })
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(OUTPUT_DIR, f'A_Stats_{DATE}.csv'), index=False)

    # Line plot — Jaccard vs config
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, method in enumerate(METHODS):
        means = res['Jaccard'][k].mean(axis=1)
        stds  = res['Jaccard'][k].std(axis=1)
        ax.errorbar(range(n_cfg), means, yerr=stds,
                    color=COLORS[method], marker=MARKERS[method],
                    linestyle=LINESTYLES[method], capsize=3, label=method)
    ax.set_xticks(range(n_cfg))
    ax.set_xticklabels(cfg_names, rotation=45, ha='right')
    ax.set_ylabel('Jaccard index')
    ax.set_title(f'Experiment A - Jaccard across 18 preprocessing configs ({n_img} frames)')
    ax.legend()
    ax.set_ylim(0, 1)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'A_Plot_Jaccard_{DATE}.{ext}'))
    plt.close(fig)

    # FPS bar
    fig, ax = plt.subplots(figsize=(6, 4))
    fps = [1.0 / res['Time_s'][k].mean() for k in range(n_meth)]
    bars = ax.bar(METHODS, fps, color=[COLORS[m] for m in METHODS])
    for b, v in zip(bars, fps):
        ax.text(b.get_x() + b.get_width()/2, v, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('FPS')
    ax.set_title('Experiment A - speed comparison')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'A_Plot_FPS_{DATE}.{ext}'))
    plt.close(fig)


# ============================================================================
# Experiment B1 — Synthetic semicircle
# ============================================================================

def run_experiment_B1():
    x0, y0, r0    = B1_X0, B1_Y0, B1_R0
    n_pts         = 50
    noise_std     = 1.0
    outlier_range = list(range(6))
    xmax, ymax, rmax = x0 + r0, y0, r0 * 2
    rmin = max(int(0.5 * r0), 5)

    n_meth = len(METHODS)
    n_out  = len(outlier_range)
    J_mean  = np.zeros((n_meth, n_out))
    J_std   = np.zeros((n_meth, n_out))
    AD_mean = np.zeros((n_meth, n_out))
    R_mean  = np.zeros((n_meth, n_out))
    T_mean  = np.zeros((n_meth, n_out))

    print(f"Experiment B1: {n_out} outlier configs x {n_meth} methods x {N_ITER_B} iters")
    print("=" * 70)
    t_start = time.time()

    for oi, n_out_pts in enumerate(outlier_range):
        J_buf  = np.zeros((n_meth, N_ITER_B))
        AD_buf = np.zeros_like(J_buf)
        R_buf  = np.zeros_like(J_buf)
        T_buf  = np.zeros_like(J_buf)
        for it in range(N_ITER_B):
            rng    = np.random.default_rng(it * 1000 + oi)
            points = generate_semicircle_points(x0, y0, r0, n_pts,
                                                noise_std=noise_std,
                                                n_outliers=n_out_pts, rng=rng)
            for k, method in enumerate(METHODS):
                cx, cy, r, elapsed = run_method_B(method, points,
                                                   xmax, ymax, rmax, rmin=rmin)
                T_buf[k, it] = elapsed
                if r > 0:
                    J_buf[k, it]  = jaccard_circles(x0, y0, r0, cx, cy, r)
                    AD_buf[k, it] = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)
                    R_buf[k, it]  = abs(r0 - r)
        J_mean[:, oi]  = J_buf.mean(axis=1)
        J_std[:, oi]   = J_buf.std(axis=1)
        AD_mean[:, oi] = AD_buf.mean(axis=1)
        R_mean[:, oi]  = R_buf.mean(axis=1)
        T_mean[:, oi]  = T_buf.mean(axis=1)
        print(f"  outliers={n_out_pts} done  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    return {'J_mean': J_mean, 'J_std': J_std, 'AD_mean': AD_mean,
            'R_mean': R_mean, 'T_mean': T_mean, 'outlier_range': outlier_range}


def save_experiment_B1(res):
    outs = res['outlier_range']

    rows = []
    for k, method in enumerate(METHODS):
        for oi, n_out in enumerate(outs):
            rows.append({
                'Method':   method,
                'Outliers': n_out,
                'J_mean':   round(res['J_mean'][k, oi],  4),
                'J_std':    round(res['J_std'][k, oi],   4),
                'AD_mean':  round(res['AD_mean'][k, oi], 4),
                'R_mean':   round(res['R_mean'][k, oi],  4),
                'Time_s':   round(res['T_mean'][k, oi],  6),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f'B1_Summary_{DATE}.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, method in enumerate(METHODS):
        axes[0].errorbar(outs, res['J_mean'][k], yerr=res['J_std'][k],
                         color=COLORS[method], marker=MARKERS[method],
                         linestyle=LINESTYLES[method], capsize=3, label=method)
    axes[0].set_xlabel('Number of outliers')
    axes[0].set_ylabel('Jaccard index')
    axes[0].set_title('B1: Jaccard vs outliers (semicircle, r=100)')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    for k, method in enumerate(METHODS):
        axes[1].plot(outs, res['AD_mean'][k],
                     color=COLORS[method], marker=MARKERS[method],
                     linestyle=LINESTYLES[method], label=method)
    axes[1].set_xlabel('Number of outliers')
    axes[1].set_ylabel('Average distance (mm)')
    axes[1].set_title('B1: Center error vs outliers')
    axes[1].legend()

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'B1_Plot_{DATE}.{ext}'))
    plt.close(fig)


# ============================================================================
# Experiment B2 — Synthetic full circle, noise x outliers x quantization
# ============================================================================

def run_experiment_B2():
    x0, y0, r0 = B2_X0, B2_Y0, B2_R0
    N           = B2_N_POINTS

    if FAST:
        noise_pct   = [0, 5]
        outlier_pct = [0, 30, 70]
        q_values    = [0, 6, 24]
    else:
        noise_pct   = [0, 1, 2, 5, 10]
        outlier_pct = [0, 10, 20, 30, 40, 50, 60, 70]
        q_values    = [0, 1, 2, 3, 6, 12, 24, 40]

    nN, nO, nQ = len(noise_pct), len(outlier_pct), len(q_values)
    n_meth     = len(METHODS)

    J_stats     = np.zeros((n_meth, nN, nO, nQ, 5))   # [min, mean, median, max, std]
    T_mean_cell = np.zeros((n_meth, nN, nO, nQ))

    total = nN * nO * nQ
    done  = 0
    t_start = time.time()
    print(f"Experiment B2: {total} configs x {n_meth} methods x {N_ITER_B} iters")
    print("=" * 70)

    for ni, np_pct in enumerate(noise_pct):
        noise_std = (np_pct / 100.0) * r0
        for oi, op in enumerate(outlier_pct):
            n_outliers = round(op / 100.0 * N)
            for qi, q in enumerate(q_values):
                if q > 0:
                    x0_q = int(x0 // q); y0_q = int(y0 // q); r0_q = int(r0 // q)
                    xmax = max(x0_q + r0_q, 10)
                    ymax = max(y0_q + r0_q, 10)
                    rmax = max(int(r0_q * 2), 5)
                    rmin = max(int(r0_q * 0.5), 4)
                else:
                    x0_q, y0_q, r0_q = x0, y0, r0
                    xmax = x0 + r0; ymax = y0 + r0; rmax = int(r0 * 2)
                    rmin = max(int(r0 * 0.5), 4)

                J_buf = np.zeros((n_meth, N_ITER_B))
                T_buf = np.zeros((n_meth, N_ITER_B))
                for it in range(N_ITER_B):
                    rng   = np.random.default_rng(it * 10000 + ni * 1000 + oi * 100 + qi)
                    pts   = generate_circle_points(x0, y0, r0, N,
                                                   noise_std=noise_std,
                                                   n_outliers=n_outliers, rng=rng)
                    pts_q = apply_quantization(pts, q)
                    if len(pts_q) < 3:
                        continue
                    for k, method in enumerate(METHODS):
                        cx, cy, r, elapsed = run_method_B(method, pts_q,
                                                           xmax, ymax, rmax,
                                                           rmin=rmin)
                        T_buf[k, it] = elapsed
                        if r > 0:
                            J_buf[k, it] = jaccard_circles(x0_q, y0_q, r0_q,
                                                           cx, cy, r)

                J_stats[:, ni, oi, qi, 0] = J_buf.min(axis=1)
                J_stats[:, ni, oi, qi, 1] = J_buf.mean(axis=1)
                J_stats[:, ni, oi, qi, 2] = np.median(J_buf, axis=1)
                J_stats[:, ni, oi, qi, 3] = J_buf.max(axis=1)
                J_stats[:, ni, oi, qi, 4] = J_buf.std(axis=1)
                T_mean_cell[:, ni, oi, qi] = T_buf.mean(axis=1)

                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  {done}/{total} configs  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    print(f"Done in {time.time()-t_start:.1f}s")
    return {'J_stats': J_stats, 'T_mean': T_mean_cell,
            'noise_pct': noise_pct, 'outlier_pct': outlier_pct,
            'q_values': q_values}


def save_experiment_B2(res):
    J_mean = res['J_stats'][..., 1]   # (n_meth, nN, nO, nQ)
    noise_pct, outlier_pct, q_values = res['noise_pct'], res['outlier_pct'], res['q_values']

    # Long-format CSV
    rows = []
    for k, method in enumerate(METHODS):
        for ni, np_ in enumerate(noise_pct):
            for oi, op in enumerate(outlier_pct):
                for qi, q in enumerate(q_values):
                    rows.append({
                        'Method':    method,
                        'Noise_pct': np_,
                        'Outlier_pct': op,
                        'q':         q,
                        'J_mean':    round(res['J_stats'][k, ni, oi, qi, 1], 4),
                        'J_std':     round(res['J_stats'][k, ni, oi, qi, 4], 4),
                        'Time_s':    round(res['T_mean'][k, ni, oi, qi], 6),
                    })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f'B2_Summary_{DATE}.csv'), index=False)

    # 3-panel plot — Jaccard vs each axis (averaged over the other two)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # vs q
    for k, method in enumerate(METHODS):
        v = J_mean[k].mean(axis=(0, 1))
        axes[0].plot(q_values, v, color=COLORS[method], marker=MARKERS[method],
                     linestyle=LINESTYLES[method], label=method)
    axes[0].set_xlabel('Quantization q')
    axes[0].set_ylabel('Jaccard (mean)')
    axes[0].set_title('B2: vs spatial quantization')
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # vs outliers
    for k, method in enumerate(METHODS):
        v = J_mean[k].mean(axis=(0, 2))
        axes[1].plot(outlier_pct, v, color=COLORS[method], marker=MARKERS[method],
                     linestyle=LINESTYLES[method], label=method)
    axes[1].set_xlabel('Outlier %')
    axes[1].set_ylabel('Jaccard (mean)')
    axes[1].set_title('B2: vs outlier contamination')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    # vs noise
    for k, method in enumerate(METHODS):
        v = J_mean[k].mean(axis=(1, 2))
        axes[2].plot(noise_pct, v, color=COLORS[method], marker=MARKERS[method],
                     linestyle=LINESTYLES[method], label=method)
    axes[2].set_xlabel('Noise sigma/r0 (%)')
    axes[2].set_ylabel('Jaccard (mean)')
    axes[2].set_title('B2: vs measurement noise')
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'B2_Plot_{DATE}.{ext}'))
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"3C-FBI v1 vs v2 comparison  -  {'FAST' if FAST else 'FULL'} mode")
    print(f"Output: {OUTPUT_DIR}/  (date tag {DATE})")
    print("=" * 70)
    t_total = time.time()

    print("\n[1/3] Experiment A - real-world data")
    res_A = run_experiment_A()
    save_experiment_A(res_A)

    print("\n[2/3] Experiment B1 - synthetic semicircle")
    res_B1 = run_experiment_B1()
    save_experiment_B1(res_B1)

    print("\n[3/3] Experiment B2 - synthetic full circle grid")
    res_B2 = run_experiment_B2()
    save_experiment_B2(res_B2)

    print("\n" + "=" * 70)
    print(f"Total runtime: {time.time()-t_total:.1f}s  ({(time.time()-t_total)/3600:.2f} h)")
    print(f"Outputs in {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
