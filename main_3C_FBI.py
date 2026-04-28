"""
main_3C_FBI.py — 3C-FBI Comprehensive Evaluation
Publication-quality outputs for journal submission.

Three experiments as described in the paper:
  A  : Real-world Parkinson's disease data (144 frames × 18 preprocessing configs)
       Methods: CIBICA, 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi
       Reports: (i) average over ALL 18 configs, (ii) best config per method,
                (iii) mean over GL80/GL82/GL84 (Table 1 style)
  B1 : Synthetic semicircle with varying outliers  (following Qi et al. 2024)
       Methods: 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi
  B2 : Synthetic full circle — varying noise, outliers, spatial quantization
       Methods: 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi

Output folder: CCC_FBI_results/

Usage:
    cd /Users/erc/Documents/3C-FBI-Circle-fitting
    conda activate poseestimation
    python main_3C_FBI.py
"""

import contextlib
import io
import math as m
import os
import time
from datetime import date
from itertools import combinations as _combinations

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from algorithms.CCC_FBI   import ccc_fbi
from algorithms.CIBICA    import CIBICA as cibica_fn
from algorithms.RHT       import rht
from algorithms.RCD       import rcd
from algorithms.RFCA      import rfca
from algorithms.NURUNNABI import nurunnabi
from algorithms.GUO       import guo_2019
from algorithms.GRECO     import greco_2022
from algorithms.QI        import qi_2024
from algorithms.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)

# ============================================================================
# Global publication-quality plot settings
# ============================================================================

plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            11,
    'axes.titlesize':       13,
    'axes.titleweight':     'bold',
    'axes.labelsize':       12,
    'axes.labelweight':     'bold',
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      9.5,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     '0.8',
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.facecolor':    'white',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.3,
    'grid.linestyle':       '--',
    'lines.linewidth':      2.0,
    'lines.markersize':     6,
    'errorbar.capsize':     3,
})

# ============================================================================
# Global constants
# ============================================================================

DATE       = date.today().strftime('%Y%m%d')
OUTPUT_DIR = 'CCC_FBI_results'

# Methods used in Experiment A (real data) — includes CIBICA
METHODS_A = ['CIBICA', '3C-FBI', 'RHT', 'RCD', 'RFCA', 'Nurunnabi', 'Guo', 'Greco', 'Qi']

# Paper view for Experiment A — excludes CIBICA (same-author predecessor, not a competitor)
METHODS_A_PAPER = ['3C-FBI', 'RHT', 'RCD', 'RFCA', 'Nurunnabi', 'Guo', 'Greco', 'Qi']

# Methods used in Experiments B1/B2 (synthetic) — CIBICA excluded (radius filter incompatible)
METHODS = ['3C-FBI', 'RHT', 'RCD', 'RFCA', 'Nurunnabi', 'Guo', 'Greco', 'Qi']

# Publication-quality color palette (colorblind-friendly)
COLORS = {
    'CIBICA':    '#2ca02c',   # green
    '3C-FBI':    '#1f77b4',   # blue  ← proposed method
    'RHT':       '#d62728',   # red
    'RCD':       '#ff7f0e',   # orange
    'RFCA':      '#9467bd',   # purple
    'Nurunnabi': '#8c564b',   # brown
    'Guo':       '#e377c2',   # pink
    'Greco':     '#7f7f7f',   # grey
    'Qi':        '#bcbd22',   # olive
}

# Marker styles — one per method for black-and-white compatibility
MARKERS = {
    'CIBICA':    's',    # square
    '3C-FBI':    'o',    # circle  ← proposed
    'RHT':       '^',    # triangle up
    'RCD':       'v',    # triangle down
    'RFCA':      'D',    # diamond
    'Nurunnabi': 'P',    # plus-filled
    'Guo':       'X',    # x-filled
    'Greco':     'h',    # hexagon
    'Qi':        '*',    # star
}

LINESTYLES = {
    'CIBICA':    (0, (3, 1, 1, 1)),   # dash-dot
    '3C-FBI':    'solid',
    'RHT':       'dashed',
    'RCD':       'dotted',
    'RFCA':      (0, (5, 2)),
    'Nurunnabi': (0, (3, 2)),
    'Guo':       (0, (1, 1)),
    'Greco':     (0, (5, 1, 1, 1)),
    'Qi':        (0, (4, 2, 1, 2)),
}

# GL thresholds used for the summary Table 1 in the paper (GL80, GL82, GL84)
BEST_GL = ['GL80', 'GL82', 'GL84']

# Synthetic experiment parameters
N_ITER_B             = 100    # Monte-Carlo iterations per configuration
B1_X0, B1_Y0, B1_R0 = 50,  60,  100
B2_X0, B2_Y0, B2_R0 = 120, 120, 120
B2_N_POINTS          = 100

# Significance stars
def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ============================================================================
# Jaccard index (analytical exact formula)
# ============================================================================

def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) between two circles."""
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d == 0:
        return min((r1/r2)**2, (r2/r1)**2)
    d1 = (d**2 + r1**2 - r2**2) / (2*d)
    d2 = d - d1
    R, r = max(r1, r2), min(r1, r2)
    if d >= r1 + r2:
        return 0.0
    elif d <= R - r:
        return (r/R)**2
    a1 = 2 * m.acos(max(-1.0, min(1.0, d1/r1)))
    a2 = 2 * m.acos(max(-1.0, min(1.0, d2/r2)))
    inter = 0.5*r1**2*(a1 - m.sin(a1)) + 0.5*r2**2*(a2 - m.sin(a2))
    union = m.pi*(R**2 + r**2) - inter
    return inter / union


# ============================================================================
# Statistical helpers
# ============================================================================

def _hl_estimator_ci(a, b, n_boot=4000, ci_level=0.95, seed=42):
    """
    Hodges-Lehmann estimator and bootstrap CI for paired differences (a - b).

    Returns
    -------
    hl : float   — HL point estimate (median of Walsh averages of differences)
    lo : float   — lower CI bound
    hi : float   — upper CI bound
    """
    diff = np.asarray(a, float) - np.asarray(b, float)
    n    = len(diff)
    ii, jj = np.triu_indices(n, k=0)
    walsh  = (diff[ii] + diff[jj]) / 2.0
    hl     = float(np.median(walsh))

    rng  = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for s in range(n_boot):
        samp   = rng.choice(diff, size=n, replace=True)
        wi, wj = np.triu_indices(n, k=0)
        boot[s]= np.median((samp[wi] + samp[wj]) / 2.0)

    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return hl, lo, hi


def _rank_biserial(w_plus, n):
    """Rank-biserial correlation from Wilcoxon W+ statistic."""
    max_w = n * (n + 1) / 2.0
    return float(2 * w_plus / max_w - 1)


def compute_focal_stats_A(res, focal='3C-FBI', methods=None):
    """
    Compute Wilcoxon + Hodges-Lehmann + 95% CI + r_rb for focal method
    vs every other method.  Scores are per-image means over the best 3 GL configs.

    Returns list of dicts, one per comparison.
    """
    if methods is None:
        methods = METHODS_A
    cfg_names = res['config_names']
    best_idx  = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]

    scores = {}
    for method in methods:
        k = METHODS_A.index(method)
        scores[method] = res['Jaccard'][k][best_idx, :].mean(axis=0)  # (144,)

    focal_scores = scores[focal]
    rows = []
    for method in methods:
        if method == focal:
            continue
        other = scores[method]
        diff  = focal_scores - other

        if np.all(diff == 0):
            row = dict(Baseline=method, HL=0.0, CI_lo=0.0, CI_hi=0.0,
                       W_stat=np.nan, p_value=np.nan, r_rb=np.nan,
                       Stars='ns', n_better=0, n_worse=0, n_tie=0)
        else:
            stat, p = wilcoxon(focal_scores, other, alternative='two-sided')
            hl, lo, hi = _hl_estimator_ci(focal_scores, other)
            n = len(diff)
            r_rb = _rank_biserial(float(stat), n)
            row = dict(
                Baseline    = method,
                HL          = round(hl,  4),
                CI_lo       = round(lo,  4),
                CI_hi       = round(hi,  4),
                W_stat      = round(float(stat), 1),
                p_value     = float(p),
                r_rb        = round(r_rb, 3),
                Stars       = _sig_stars(p),
                n_better    = int(np.sum(diff > 0)),
                n_worse     = int(np.sum(diff < 0)),
                n_tie       = int(np.sum(diff == 0)),
            )
        rows.append(row)
    return rows


# ============================================================================
# Method runners
# ============================================================================

def _call_ccc_fbi(edgels, xmax, ymax, rmax=40):
    """Suppress the internal print() in ccc_fbi."""
    with contextlib.redirect_stdout(io.StringIO()):
        center, r = ccc_fbi(edgels, Nmax=5000, xmax=xmax, ymax=ymax, rmax=rmax)
    return np.array(center, dtype=float), float(r)


def run_method_A(method, edgels, xmax, ymax):
    """Run one method on real-image edgels (Experiment A).
    Edgels: shape (N,2), [row, col]. Returns center [row,col], r, elapsed."""
    t0 = time.perf_counter()
    try:
        if method == 'CIBICA':
            x_c, y_c, r = cibica_fn(edgels, n_triplets=500, xmax=xmax, ymax=ymax)
            center = np.array([y_c, x_c], dtype=float)
            r = float(r)
        elif method == '3C-FBI':    center, r = _call_ccc_fbi(edgels, xmax, ymax)
        elif method == 'RHT':       center, r = rht(edgels, num_iterations=1000, threshold=3)
        elif method == 'RCD':       center, r = rcd(edgels, num_iterations=1000,
                                                    distance_threshold=2,
                                                    min_inliers=5, min_distance=5)
        elif method == 'RFCA':      center, r = rfca(edgels)
        elif method == 'Nurunnabi': center, r = nurunnabi(edgels)
        elif method == 'Guo':       center, r = guo_2019(edgels)
        elif method == 'Greco':     center, r = greco_2022(edgels)
        elif method == 'Qi':        center, r = qi_2024(edgels)
        else:                       return np.array([-1., -1.]), -1., 0.
    except Exception:
        return np.array([-1., -1.]), -1., time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    center  = np.array(center, dtype=float)
    r       = float(r)
    if r <= 0 or np.any(np.isnan(center)):
        return np.array([-1., -1.]), -1., elapsed
    return center, r, elapsed


def run_method_B(method, points, xmax, ymax, rmax=300):
    """Run one method on a synthetic point cloud (Experiments B1/B2).
    Points in (x,y) space. Returns cx, cy, r, elapsed."""
    t0 = time.perf_counter()
    try:
        if   method == '3C-FBI':    center, r = _call_ccc_fbi(points, xmax, ymax, rmax=rmax)
        elif method == 'RHT':       center, r = rht(points, num_iterations=1000, threshold=5)
        elif method == 'RCD':       center, r = rcd(points, num_iterations=1000,
                                                    distance_threshold=2,
                                                    min_inliers=5, min_distance=5)
        elif method == 'RFCA':      center, r = rfca(points)
        elif method == 'Nurunnabi': center, r = nurunnabi(points)
        elif method == 'Guo':       center, r = guo_2019(points)
        elif method == 'Greco':     center, r = greco_2022(points)
        elif method == 'Qi':        center, r = qi_2024(points)
        else:                       return np.nan, np.nan, -1., 0.
    except Exception:
        return np.nan, np.nan, -1., time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    cx, cy  = float(center[0]), float(center[1])
    r       = float(r)
    if r <= 0 or np.isnan(cx) or np.isnan(cy):
        return np.nan, np.nan, -1., elapsed
    return cx, cy, r, elapsed


# ============================================================================
# Experiment A — Real-world clinical data
# ============================================================================

def run_experiment_A():
    """144 frames × 18 preprocessing configs × 9 methods (incl. CIBICA).

    Uses the same data pipeline as main_CIBICA_2026.py:
      - Same Ground_Truth.csv, same ROI images
      - Same 18 preprocessing configs from algorithms/preprocessing.py
      - Same method parameters (RHT threshold=3, RCD dist=2/inl=5/dist=5, etc.)
    All 9 methods run on every (image, config) pair.
    """
    gt       = pd.read_csv('data/Ground_Truth.csv')
    files    = gt['Filename'].tolist()
    XGT_arr  = gt['X'].to_numpy()
    YGT_arr  = gt['Y'].to_numpy()
    RGT_arr  = gt['R'].to_numpy()
    configs  = get_preprocessing_configs()
    cfg_names = [c['name'] for c in configs]

    n_img  = len(files)
    n_cfg  = len(configs)
    n_meth = len(METHODS_A)

    Jaccard = np.zeros((n_meth, n_cfg, n_img))
    AD      = np.zeros_like(Jaccard)
    RMSE    = np.zeros_like(Jaccard)
    Time_s  = np.zeros_like(Jaccard)

    print(f"Experiment A: {n_img} images × {n_cfg} configs × {n_meth} methods")
    print("=" * 70)
    t_start = time.time()

    for i, filename in enumerate(files):
        XGT, YGT, RGT = XGT_arr[i], YGT_arr[i], RGT_arr[i]
        BS_crop = cv2.imread(os.path.join('data', 'black_sphere_ROI', filename + '.png'))
        G_crop  = cv2.imread(os.path.join('data', 'green_back_ROI',   filename + '.png'))
        if BS_crop is None:
            print(f"  Warning: missing {filename} — skipping")
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

            for k, method in enumerate(METHODS_A):
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
    """Save CSVs and publication-quality figures for Experiment A.

    Produces two complete sets of outputs:
      tag='A'  — all 9 methods (with CIBICA)
      tag='Ap' — paper view, 8 methods (without CIBICA)
    Raw per-method CSVs and timing are saved once for all methods.
    """
    cfg_names = res['config_names']

    # ── Raw per-method CSV (once, all methods) ────────────────────────────────
    for k, method in enumerate(METHODS_A):
        df = pd.DataFrame(res['Jaccard'][k].T,
                          index=res['filenames'], columns=cfg_names)
        df.index.name = 'Filename'
        df.to_csv(os.path.join(OUTPUT_DIR, f'A_Jaccard_{method}_{DATE}.csv'))

    # Raw timing CSV (once, all methods)
    timing_rows = []
    for k, method in enumerate(METHODS_A):
        for j, cfg in enumerate(cfg_names):
            t_mean = res['Time_s'][k, j, :].mean()
            timing_rows.append({'Method': method, 'Config': cfg,
                                 'Time_s': round(t_mean, 6),
                                 'FPS': round(1.0 / t_mean if t_mean > 0 else 0, 1)})
    pd.DataFrame(timing_rows).to_csv(
        os.path.join(OUTPUT_DIR, f'A_Timing_Raw_{DATE}.csv'), index=False)
    print(f"  Saved: A_Timing_Raw_{DATE}.csv")

    # ── Two views: with and without CIBICA ────────────────────────────────────
    for methods, tag in [(METHODS_A, 'A'), (METHODS_A_PAPER, 'Ap')]:
        print(f"\n  --- View '{tag}': {len(methods)} methods ---")
        _save_A_view(res, methods, tag)


def _save_A_view(res, methods, tag):
    """Save summary tables, statistics, and all figures for one method set."""
    cfg_names = res['config_names']
    n_meth    = len(methods)
    n_cfg     = len(cfg_names)
    best_idx  = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    kidx      = [METHODS_A.index(m) for m in methods]

    # ── Helper: build summary row ─────────────────────────────────────────────
    def _row(k_full, cfg_idx):
        J    = res['Jaccard'][k_full][cfg_idx, :].mean()
        AD_v = res['AD']    [k_full][cfg_idx, :].mean()
        RM   = res['RMSE']  [k_full][cfg_idx, :].mean()
        t    = res['Time_s'][k_full][cfg_idx, :].mean()
        fps  = 1.0 / t if t > 0 else 0
        J_s  = res['Jaccard'][k_full][cfg_idx, :].std()
        return round(J, 4), round(J_s, 4), round(AD_v, 3), round(RM, 3), round(fps, 1)

    # ── Summary tables ────────────────────────────────────────────────────────
    # (i) All configs
    rows_all = []
    for i, method in enumerate(methods):
        J, Js, AD_v, RM, fps = _row(kidx[i], np.arange(n_cfg))
        rows_all.append({'Method': method, 'Jaccard_mean': J, 'Jaccard_std': Js,
                         'AD_px': AD_v, 'RMSE_px': RM, 'FPS': fps})
    df_all = pd.DataFrame(rows_all).set_index('Method')
    df_all.to_csv(os.path.join(OUTPUT_DIR, f'{tag}_Table_AllConfigs_{DATE}.csv'))
    print(f"  Saved: {tag}_Table_AllConfigs_{DATE}.csv")

    # (ii) Best config per method
    rows_best, best_cfg_per_method = [], []
    for i, method in enumerate(methods):
        best_j = int(np.argmax(res['Jaccard'][kidx[i]].mean(axis=1)))
        best_cfg_per_method.append(best_j)
        J, Js, AD_v, RM, fps = _row(kidx[i], best_j)
        rows_best.append({'Method': method, 'Best_Config': cfg_names[best_j],
                          'Jaccard_mean': J, 'Jaccard_std': Js,
                          'AD_px': AD_v, 'RMSE_px': RM, 'FPS': fps})
    df_best = pd.DataFrame(rows_best).set_index('Method')
    df_best.to_csv(os.path.join(OUTPUT_DIR, f'{tag}_Table_BestConfig_{DATE}.csv'))
    print(f"  Saved: {tag}_Table_BestConfig_{DATE}.csv")

    # (iii) GL80/GL82/GL84
    rows_gl = []
    for i, method in enumerate(methods):
        J, Js, AD_v, RM, fps = _row(kidx[i], best_idx)
        rows_gl.append({'Method': method, 'Jaccard_mean': J, 'Jaccard_std': Js,
                        'AD_px': AD_v, 'RMSE_px': RM, 'FPS': fps})
    df_gl = pd.DataFrame(rows_gl).set_index('Method')
    df_gl.to_csv(os.path.join(OUTPUT_DIR, f'{tag}_Table1_Best3GL_{DATE}.csv'))
    print(f"  Saved: {tag}_Table1_Best3GL_{DATE}.csv")

    # ── Statistical analysis: 3C-FBI vs each baseline ────────────────────────
    focal_stats = compute_focal_stats_A(res, focal='3C-FBI', methods=methods)
    df_stat = pd.DataFrame(focal_stats)
    df_stat.to_csv(os.path.join(OUTPUT_DIR, f'{tag}_Stats_FocalTest_{DATE}.csv'), index=False)
    print(f"  Saved: {tag}_Stats_FocalTest_{DATE}.csv")

    # Pairwise all-vs-all Wilcoxon
    scores = {}
    for i, m in enumerate(methods):
        scores[m] = res['Jaccard'][kidx[i]][best_idx, :].mean(axis=0)
    pairs  = [(a, b) for i, a in enumerate(methods) for b in methods[i+1:]]
    pw_rows = []
    for a, b in pairs:
        diff = scores[a] - scores[b]
        if np.all(diff == 0):
            stat, p = np.nan, np.nan
        else:
            stat, p = wilcoxon(scores[a], scores[b], alternative='two-sided')
        pw_rows.append({'Method_A': a, 'Method_B': b,
                        'Mean_A': round(float(np.mean(scores[a])), 4),
                        'Mean_B': round(float(np.mean(scores[b])), 4),
                        'Delta':  round(float(np.mean(scores[a]) - np.mean(scores[b])), 4),
                        'W_stat': stat, 'p_value': p,
                        'Stars':  _sig_stars(p) if p is not np.nan and not np.isnan(p) else 'ns'})
    pd.DataFrame(pw_rows).to_csv(os.path.join(OUTPUT_DIR, f'{tag}_Stats_Pairwise_{DATE}.csv'), index=False)
    print(f"  Saved: {tag}_Stats_Pairwise_{DATE}.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Line plot: Jaccard vs all 18 preprocessing configs
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(n_cfg)
    for i, method in enumerate(methods):
        mean_j = res['Jaccard'][kidx[i]].mean(axis=1)
        ax.plot(x, mean_j,
                color=COLORS[method], linewidth=2.0,
                marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], label=method, zorder=3)
    ax.axvspan(-0.5, 8.5, alpha=0.04, color='steelblue', label='_nolegend_')
    ax.axvspan(8.5, n_cfg - 0.5, alpha=0.04, color='darkorange', label='_nolegend_')
    ax.axvline(x=8.5, color='0.5', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(4, 0.55, 'Green-Level', ha='center', va='bottom', fontsize=9,
            color='steelblue', alpha=0.8)
    ax.text(13.5, 0.55, 'Median Filter', ha='center', va='bottom', fontsize=9,
            color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(-0.5, n_cfg - 0.5)
    ax.set_xlabel('Preprocessing Configuration')
    ax.set_ylabel('Mean Jaccard Index')
    ax.set_title('Experiment A — Mean Jaccard Index across 18 Preprocessing Configurations\n'
                 f'(144 clinical frames, {n_meth} methods)')
    ax.legend(loc='lower left', ncol=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{tag}_Fig1_Jaccard_AllConfigs_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — Heatmap: methods × 18 configs (Jaccard values)
    # ══════════════════════════════════════════════════════════════════════════
    J_matrix = np.array([res['Jaccard'][kidx[i]].mean(axis=1) for i in range(n_meth)])

    fig, ax = plt.subplots(figsize=(15, max(3.5, n_meth * 0.5)))
    cmap = LinearSegmentedColormap.from_list('jac', ['#d73027','#fee090','#4575b4'], N=256)
    vmin, vmax = max(0.5, J_matrix.min() - 0.01), min(1.0, J_matrix.max() + 0.005)
    im = ax.imshow(J_matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cbar.set_label('Mean Jaccard Index', fontsize=10)
    for i in range(n_meth):
        for j in range(n_cfg):
            val  = J_matrix[i, j]
            col  = 'white' if val < (vmin + vmax) / 2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6.5,
                    color=col, fontweight='bold')
    ax.set_xticks(np.arange(n_cfg))
    ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticks(np.arange(n_meth))
    ax.set_yticklabels(methods, fontsize=10)
    for i in range(n_meth):
        best_j = int(np.argmax(J_matrix[i]))
        rect = plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor='gold', linewidth=2.5)
        ax.add_patch(rect)
    ax.axvline(8.5, color='white', linewidth=1.5, alpha=0.6)
    ax.set_title('Experiment A — Jaccard Index Heatmap (Methods × Preprocessing Configs)\n'
                 'Gold border = best config per method', pad=8)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{tag}_Fig2_Heatmap_MethodxConfig_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 3 — Violin + strip plot at GL82 reference
    # ══════════════════════════════════════════════════════════════════════════
    best_single = cfg_names.index('GL82') if 'GL82' in cfg_names else best_idx[0]
    _plot_violin_strip(
        [res['Jaccard'][kidx[i]][best_single, :] for i in range(n_meth)],
        methods,
        title=f'Experiment A — Jaccard Distribution at GL82 (144 frames)',
        ylabel='Jaccard Index',
        path=os.path.join(OUTPUT_DIR, f'{tag}_Fig3_Violin_GL82_{DATE}.png'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 4 — Violin + strip at best config per method
    # ══════════════════════════════════════════════════════════════════════════
    data_bc   = [res['Jaccard'][kidx[i]][best_cfg_per_method[i], :] for i in range(n_meth)]
    labels_bc = [f"{m}\n({cfg_names[best_cfg_per_method[i]]})"
                 for i, m in enumerate(methods)]
    _plot_violin_strip(
        data_bc, methods,
        tick_labels=labels_bc,
        title='Experiment A — Jaccard at Best Config per Method (144 frames)',
        ylabel='Jaccard Index',
        path=os.path.join(OUTPUT_DIR, f'{tag}_Fig4_Violin_BestConfig_{DATE}.png'),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 5 — Focal statistical test: 3C-FBI vs baselines
    # ══════════════════════════════════════════════════════════════════════════
    _plot_focal_stats(focal_stats,
                      path=os.path.join(OUTPUT_DIR, f'{tag}_Fig5_Stats_FocalTest_{DATE}.png'))

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 6 — FPS comparison bar chart
    # ══════════════════════════════════════════════════════════════════════════
    fps_vals = []
    for i in range(n_meth):
        t = res['Time_s'][kidx[i]][best_idx, :].mean()
        fps_vals.append(1.0 / t if t > 0 else 0)

    fig, ax = plt.subplots(figsize=(9, max(4, n_meth * 0.55)))
    bars = ax.barh(methods[::-1], fps_vals[::-1],
                   color=[COLORS[m] for m in methods[::-1]],
                   edgecolor='white', linewidth=0.5, height=0.65)
    for bar, val in zip(bars, fps_vals[::-1]):
        ax.text(val + max(fps_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f} fps', va='center', fontsize=9.5, fontweight='bold')
    ax.axvline(30, color='crimson', linestyle='--', linewidth=1.5,
               label='Real-time threshold (30 fps)', alpha=0.8)
    ax.set_xlabel('Frames per Second (FPS) — higher is better')
    ax.set_title('Experiment A — Computational Speed per Method\n'
                 '(averaged over GL80/GL82/GL84, 144 frames)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(fps_vals) * 1.18)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{tag}_Fig6_FPS_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 7 — Pairwise Wilcoxon p-value heatmap
    # ══════════════════════════════════════════════════════════════════════════
    n  = len(methods)
    pmat = np.ones((n, n))
    for row in pw_rows:
        i = methods.index(row['Method_A'])
        j = methods.index(row['Method_B'])
        p_val = row['p_value']
        pmat[i, j] = pmat[j, i] = p_val if not np.isnan(float(p_val) if p_val is not None else np.nan) else 1.0

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9), max(6, n * 0.85)))
    cmap_p  = LinearSegmentedColormap.from_list('pval', ['#2166ac','#92c5de','#f4a582','#d6604d'], N=256)
    im  = ax.imshow(np.log10(pmat + 1e-10), vmin=-4, vmax=0, cmap=cmap_p)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('log₁₀(p-value)', fontsize=10)
    cbar.set_ticks([-4, -3, -2, -1, 0])
    cbar.set_ticklabels(['0.0001', '0.001', '0.01', '0.1', '1.0'])
    ax.set_xticks(range(n)); ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Experiment A — Pairwise Wilcoxon Signed-Rank p-values\n'
                 '(blue = highly significant, red = not significant)', pad=8)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='0.4')
            else:
                pv  = pmat[i, j]
                txt = _sig_stars(pv) if not np.isnan(pv) else 'ns'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if pv < 0.01 else 'black', fontweight='bold')
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{tag}_Fig7_Pairwise_Wilcoxon_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 8 — Summary panel: Jaccard + AD + FPS side by side (Table figure)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [
        ('Jaccard_mean', 'Mean Jaccard Index', df_gl['Jaccard_mean'], True),
        ('AD_px',        'Mean AD (pixels)',    df_gl['AD_px'],        False),
        ('FPS',          'Frames per Second',   df_gl['FPS'],          True),
    ]
    for ax, (col, ylabel, vals, higher_better) in zip(axes, metrics):
        colors = [COLORS[m] for m in methods]
        bars = ax.bar(range(n_meth), vals.values, color=colors,
                      edgecolor='white', linewidth=0.5, width=0.7)
        best_i = int(np.argmax(vals) if higher_better else np.argmin(vals))
        bars[best_i].set_edgecolor('gold')
        bars[best_i].set_linewidth(2.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.3f}' if col != 'FPS' else f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(n_meth))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{"↑ higher = better" if higher_better else "↓ lower = better"}',
                     fontsize=9, color='0.5')
        ax.set_xlim(-0.6, n_meth - 0.4)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle(f'Experiment A — Performance Summary (GL80/GL82/GL84, 144 frames, {n_meth} methods)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{tag}_Fig8_Summary_Panel_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_violin_strip(data_list, method_list, title, ylabel,
                       path, tick_labels=None):
    """Publication-quality violin + strip + median plot."""
    n = len(method_list)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 5.5))
    positions = np.arange(n)

    for i, (vals, method) in enumerate(zip(data_list, method_list)):
        v = np.asarray(vals)
        # Violin
        vp = ax.violinplot(v, positions=[i], widths=0.7, showmedians=False,
                           showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(COLORS[method])
            body.set_alpha(0.35)
            body.set_edgecolor(COLORS[method])
        # Box (IQR)
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q25, q75, color=COLORS[method], linewidth=5, alpha=0.6)
        # Median
        ax.scatter(i, med, color='white', s=45, zorder=5,
                   edgecolors=COLORS[method], linewidth=1.5)
        # Strip (jittered points)
        jitter = np.random.default_rng(42+i).uniform(-0.15, 0.15, len(v))
        ax.scatter(i + jitter, v, color=COLORS[method], alpha=0.25,
                   s=12, zorder=3, linewidths=0)
        # Mean diamond
        ax.scatter(i, np.mean(v), marker='D', color=COLORS[method],
                   s=40, zorder=6, edgecolors='white', linewidth=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels if tick_labels else method_list,
                       fontsize=9, rotation=15 if tick_labels else 0, ha='right')
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(max(0, min(np.concatenate(data_list)) - 0.05), 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Legend proxies
    patches = [mpatches.Patch(facecolor=COLORS[m], alpha=0.6, label=m)
               for m in method_list]
    ax.legend(handles=patches, loc='lower right', ncol=3, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")


def _plot_focal_stats(focal_stats, path):
    """Lollipop chart: HL estimate ± 95% CI for 3C-FBI vs each baseline."""
    n    = len(focal_stats)
    methods = [r['Baseline']  for r in focal_stats]
    hls     = [r['HL']        for r in focal_stats]
    lo      = [r['CI_lo']     for r in focal_stats]
    hi      = [r['CI_hi']     for r in focal_stats]
    stars   = [r['Stars']     for r in focal_stats]
    r_rbs   = [r['r_rb']      for r in focal_stats]
    pvals   = [r['p_value']   for r in focal_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(n)

    # CI bars
    for i in range(n):
        col = '#2166ac' if hls[i] > 0 else '#d6604d'
        ax.plot([lo[i], hi[i]], [i, i], color=col, linewidth=2.5,
                solid_capstyle='round', zorder=2)
        ax.scatter(hls[i], i, color=col, s=90, zorder=4,
                   edgecolors='white', linewidth=1.0)
        # Star annotation
        x_txt = hi[i] + abs(max(hi) - min(lo)) * 0.03
        ax.text(x_txt, i, f'{stars[i]}  |r_rb|={abs(r_rbs[i]):.2f}',
                va='center', fontsize=9, color=col, fontweight='bold')

    ax.axvline(0, color='0.3', linewidth=1.2, linestyle='--', alpha=0.7,
               label='No difference')
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Hodges-Lehmann Δ Jaccard (3C-FBI − Baseline)\n'
                  'with 95% bootstrap CI')
    ax.set_title('Statistical Comparison: 3C-FBI vs Baseline Methods\n'
                 '(Wilcoxon signed-rank, GL80/GL82/GL84, n=144 frames)',
                 pad=8)
    ax.legend(fontsize=9, loc='lower right')
    # Shade positive region
    xlims = ax.get_xlim()
    ax.axvspan(0, xlims[1], alpha=0.04, color='#2166ac')
    ax.axvspan(xlims[0], 0, alpha=0.04, color='#d6604d')
    ax.set_xlim(xlims)
    plt.tight_layout()
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")


def print_summary_A(res):
    for methods, label in [(METHODS_A, 'All 9 methods'),
                           (METHODS_A_PAPER, 'Paper 8 methods (no CIBICA)')]:
        _print_summary_A_view(res, methods, label)


def _print_summary_A_view(res, methods, label):
    cfg_names = res['config_names']
    best_idx  = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    kidx = [METHODS_A.index(m) for m in methods]
    hdr = f"{'Method':<10}  {'Jaccard':>8}  {'AD(px)':>8}  {'RMSE(px)':>9}  {'FPS':>7}"

    print(f"\n{'=' * 70}")
    print(f"Experiment A [{label}] — (i) Average over ALL 18 preprocessing configs")
    print("=" * 70)
    print(hdr); print("-" * 70)
    for k in kidx:
        method = METHODS_A[k]
        J  = res['Jaccard'][k].mean()
        AD = res['AD'][k].mean()
        RM = res['RMSE'][k].mean()
        t  = res['Time_s'][k].mean()
        fps = 1.0 / t if t > 0 else 0
        mark = ' ★' if method == '3C-FBI' else ''
        print(f"{method:<10}  {J:>8.3f}  {AD:>8.3f}  {RM:>9.3f}  {fps:>7.1f}{mark}")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print(f"Experiment A [{label}] — (ii) Best preprocessing config per method")
    print("=" * 70)
    print(f"{'Method':<10}  {'BestCfg':<8}  {'Jaccard':>8}  {'AD(px)':>8}  {'RMSE(px)':>9}  {'FPS':>7}")
    print("-" * 70)
    for k in kidx:
        method = METHODS_A[k]
        best_j = int(np.argmax(res['Jaccard'][k].mean(axis=1)))
        J  = res['Jaccard'][k][best_j, :].mean()
        AD = res['AD'][k][best_j, :].mean()
        RM = res['RMSE'][k][best_j, :].mean()
        t  = res['Time_s'][k][best_j, :].mean()
        fps = 1.0 / t if t > 0 else 0
        mark = ' ★' if method == '3C-FBI' else ''
        print(f"{method:<10}  {cfg_names[best_j]:<8}  {J:>8.3f}  {AD:>8.3f}  {RM:>9.3f}  {fps:>7.1f}{mark}")
    print("=" * 70)

    print(f"\n{'=' * 70}")
    print(f"Experiment A [{label}] — (iii) Mean over GL80, GL82, GL84 (Table 1 style)")
    print("=" * 70)
    print(hdr); print("-" * 70)
    for k in kidx:
        method = METHODS_A[k]
        J  = res['Jaccard'][k][best_idx, :].mean()
        AD = res['AD'][k][best_idx, :].mean()
        RM = res['RMSE'][k][best_idx, :].mean()
        t  = res['Time_s'][k][best_idx, :].mean()
        fps = 1.0 / t if t > 0 else 0
        mark = ' ★' if method == '3C-FBI' else ''
        print(f"{method:<10}  {J:>8.3f}  {AD:>8.3f}  {RM:>9.3f}  {fps:>7.1f}{mark}")
    print("=" * 70)


# ============================================================================
# Synthetic data generators
# ============================================================================

def generate_semicircle_points(x0=50, y0=60, r0=100, n_points=50,
                               noise_std=1.0, n_outliers=0, rng=None):
    """Upper semicircle (θ ∈ [0,π]) with radial noise; on-arc outliers ±9–10σ."""
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
    """Full circle with radial noise; on-arc outliers ±5–20σ."""
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
    """Spatial quantization round(x/q); q=0 → no-op."""
    if q == 0:
        return points
    return np.unique(np.round(points / q), axis=0)


# ============================================================================
# Experiment B1 — Semicircle with varying outliers
# ============================================================================

def run_experiment_B1():
    """Semicircle center (50,60), r=100 mm, n=50 pts, σ=1 mm, outliers 0-5."""
    x0, y0, r0    = B1_X0, B1_Y0, B1_R0
    n_pts         = 50
    noise_std     = 1.0
    outlier_range = list(range(6))
    xmax, ymax, rmax = x0 + r0, y0, r0 * 2

    n_meth = len(METHODS)
    n_out  = len(outlier_range)
    J_mean  = np.zeros((n_meth, n_out))
    J_std   = np.zeros((n_meth, n_out))
    AD_mean = np.zeros((n_meth, n_out))
    R_mean  = np.zeros((n_meth, n_out))
    T_mean  = np.zeros((n_meth, n_out))   # mean elapsed time (s)

    print(f"Experiment B1: {n_out} outlier configs × {n_meth} methods × {N_ITER_B} iters")
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
                cx, cy, r, elapsed = run_method_B(method, points, xmax, ymax, rmax)
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
        print(f"  Outliers={n_out_pts} done  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    return {'J_mean': J_mean, 'J_std': J_std, 'AD_mean': AD_mean,
            'R_mean': R_mean, 'T_mean': T_mean, 'outlier_range': outlier_range}


def save_experiment_B1(res):
    n_meth = len(METHODS)
    outs   = res['outlier_range']

    # ── Tables ────────────────────────────────────────────────────────────────
    for tag, arr in [('Jaccard', res['J_mean']),
                     ('AD_mm',   res['AD_mean']),
                     ('RMSE_mm', res['R_mean'])]:
        rows = []
        for k, method in enumerate(METHODS):
            row = {'Method': method}
            for oi, n_out in enumerate(outs):
                row[f'{n_out} outliers'] = round(arr[k, oi], 4)
            row['Mean'] = round(arr[k].mean(), 4)
            rows.append(row)
        path = os.path.join(OUTPUT_DIR, f'B1_{tag}_{DATE}.csv')
        pd.DataFrame(rows).set_index('Method').to_csv(path)
        print(f"  Saved: {path}")

    # Timing table: mean elapsed (s) and FPS per method × outlier count
    timing_rows = []
    for k, method in enumerate(METHODS):
        row = {'Method': method}
        for oi, n_out in enumerate(outs):
            t = res['T_mean'][k, oi]
            row[f'{n_out}out_s']   = round(t, 6)
            row[f'{n_out}out_fps'] = round(1.0 / t if t > 0 else 0, 1)
        row['Mean_FPS'] = round(1.0 / res['T_mean'][k].mean()
                                if res['T_mean'][k].mean() > 0 else 0, 1)
        timing_rows.append(row)
    pd.DataFrame(timing_rows).set_index('Method').to_csv(
        os.path.join(OUTPUT_DIR, f'B1_Timing_{DATE}.csv'))
    print(f"  Saved: B1_Timing_{DATE}.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B1-1 — Jaccard vs outlier count (main result)
    # ══════════════════════════════════════════════════════════════════════════
    x = np.array(outs)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, method in enumerate(METHODS):
        mean = res['J_mean'][k]
        std  = res['J_std'][k]
        ax.plot(x, mean, color=COLORS[method], linewidth=2.0,
                marker=MARKERS[method], markersize=6,
                linestyle=LINESTYLES[method], label=method, zorder=3)
        ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=COLORS[method], alpha=0.10)
    ax.set_xlabel('Number of Outlier Points')
    ax.set_ylabel('Mean Jaccard Index  ±1σ')
    ax.set_xticks(x)
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f'Experiment B1 — Robustness to Outliers (Semicircle)\n'
                 f'Center=({B1_X0},{B1_Y0}), r={B1_R0}, n=50, σ=1 mm, '
                 f'{N_ITER_B} trials per condition')
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B1_Fig1_Jaccard_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B1-2 — Three-panel: Jaccard + AD + RMSE
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panel_data = [
        (res['J_mean'],  res['J_std'],   'Mean Jaccard Index ±1σ',   (0.3, 1.02)),
        (res['AD_mean'], None,            'Mean Center Error (mm)',   None),
        (res['R_mean'],  None,            'Mean Radius Error (mm)',   None),
    ]
    for ax, (arr, std_arr, ylabel, ylim) in zip(axes, panel_data):
        for k, method in enumerate(METHODS):
            ax.plot(x, arr[k], color=COLORS[method], linewidth=2.0,
                    marker=MARKERS[method], markersize=5,
                    linestyle=LINESTYLES[method], label=method)
            if std_arr is not None:
                ax.fill_between(x,
                                np.clip(arr[k] - std_arr[k], 0, 1),
                                np.clip(arr[k] + std_arr[k], 0, 1),
                                color=COLORS[method], alpha=0.08)
        ax.set_xlabel('Number of Outlier Points')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_title('Jaccard Index')
    axes[1].set_title('Center Error (AD)')
    axes[2].set_title('Radius Error (RMSE)')
    axes[0].legend(ncol=2, fontsize=8.5, loc='lower left')
    fig.suptitle('Experiment B1 — Semicircle Robustness Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B1_Fig2_Panel_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")


def print_summary_B1(res):
    print("\n" + "=" * 70)
    print("Experiment B1 — Jaccard Index (mean over trials)")
    print("=" * 70)
    header = f"{'Method':<10}" + "".join(f"  {n:>4}out" for n in res['outlier_range']) + "   Mean"
    print(header); print("-" * 70)
    for k, method in enumerate(METHODS):
        vals = "".join(f"  {res['J_mean'][k, oi]:>7.3f}"
                       for oi in range(len(res['outlier_range'])))
        mark = ' ★' if method == '3C-FBI' else ''
        print(f"{method:<10}{vals}  {res['J_mean'][k].mean():>7.3f}{mark}")
    print("=" * 70)


# ============================================================================
# Experiment B2 — Full circle: noise × outliers × quantization
# ============================================================================

def run_experiment_B2():
    """
    Full circle: center (120,120), r=120 mm, N=100 pts.
    Noise σ/r₀ ∈ {0,1,2,5,10}%, outlier% ∈ {0…70}, q ∈ {0,1,2,3,6,12,24,40}.
    100 Monte-Carlo iterations. Stores [min,mean,median,max,std] per cell.
    """
    x0, y0, r0 = B2_X0, B2_Y0, B2_R0
    N           = B2_N_POINTS

    noise_pct   = [0, 1, 2, 5, 10]
    outlier_pct = [0, 10, 20, 30, 40, 50, 60, 70]
    q_values    = [0, 1, 2, 3, 6, 12, 24, 40]

    nN, nO, nQ = len(noise_pct), len(outlier_pct), len(q_values)
    n_meth      = len(METHODS)

    # [min, mean, median, max, std]
    J_stats     = np.zeros((n_meth, nN, nO, nQ, 5))
    T_mean_cell = np.zeros((n_meth, nN, nO, nQ))   # mean elapsed (s) per cell

    total  = nN * nO * nQ
    done   = 0
    t_start= time.time()

    print(f"Experiment B2: {total} configs × {n_meth} methods × {N_ITER_B} iters")
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
                else:
                    x0_q, y0_q, r0_q = x0, y0, r0
                    xmax = x0 + r0; ymax = y0 + r0; rmax = int(r0 * 2)

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
                        cx, cy, r, elapsed = run_method_B(method, pts_q, xmax, ymax, rmax)
                        T_buf[k, it] = elapsed
                        if r > 0:
                            J_buf[k, it] = jaccard_circles(x0_q, y0_q, r0_q, cx, cy, r)

                J_stats[:, ni, oi, qi, 0] = J_buf.min(axis=1)
                J_stats[:, ni, oi, qi, 1] = J_buf.mean(axis=1)
                J_stats[:, ni, oi, qi, 2] = np.median(J_buf, axis=1)
                J_stats[:, ni, oi, qi, 3] = J_buf.max(axis=1)
                J_stats[:, ni, oi, qi, 4] = J_buf.std(axis=1)
                T_mean_cell[:, ni, oi, qi] = T_buf.mean(axis=1)

                done += 1
                if done % 40 == 0 or done == total:
                    print(f"  {done}/{total} configs  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    print(f"Done in {time.time()-t_start:.1f}s")
    return {'J_stats': J_stats, 'T_mean': T_mean_cell,
            'noise_pct': noise_pct, 'outlier_pct': outlier_pct, 'q_values': q_values}


def save_experiment_B2(res):
    J_stats     = res['J_stats']        # (n_meth, nN, nO, nQ, 5)
    J_mean      = J_stats[..., 1]       # (n_meth, nN, nO, nQ)
    J_std       = J_stats[..., 4]
    noise_pct   = res['noise_pct']
    outlier_pct = res['outlier_pct']
    q_values    = res['q_values']
    n_meth      = len(METHODS)
    nN, nO, nQ  = len(noise_pct), len(outlier_pct), len(q_values)

    # ── Win count table ───────────────────────────────────────────────────────
    best        = np.argmax(J_mean, axis=0)   # (nN, nO, nQ)
    total_cells = nN * nO * nQ
    win_rows = [{'Method': m,
                 'Wins':   int(np.sum(best == k)),
                 'Pct':    round(100 * np.sum(best == k) / total_cells, 1)}
                for k, m in enumerate(METHODS)]
    df_win = pd.DataFrame(win_rows).set_index('Method')
    path = os.path.join(OUTPUT_DIR, f'B2_Table3_WinCount_{DATE}.csv')
    df_win.to_csv(path); print(f"  Saved: {path}")

    # Timing summary: mean FPS per method averaged over all configs
    T_all = res['T_mean']   # (n_meth, nN, nO, nQ)
    timing_rows = [{'Method': m,
                    'Mean_time_s': round(float(T_all[k].mean()), 6),
                    'Mean_FPS':    round(1.0 / T_all[k].mean() if T_all[k].mean() > 0 else 0, 1),
                    'Noise0_FPS':  round(1.0 / T_all[k, noise_pct.index(0)].mean()
                                         if T_all[k, noise_pct.index(0)].mean() > 0 else 0, 1)}
                   for k, m in enumerate(METHODS)]
    pd.DataFrame(timing_rows).set_index('Method').to_csv(
        os.path.join(OUTPUT_DIR, f'B2_Timing_{DATE}.csv'))
    print(f"  Saved: B2_Timing_{DATE}.csv")

    # ── Full stats CSV ────────────────────────────────────────────────────────
    stat_names = ['Min', 'Mean', 'Median', 'Max', 'Std']
    flat_rows = []
    for k, method in enumerate(METHODS):
        for ni, np_p in enumerate(noise_pct):
            for oi, op in enumerate(outlier_pct):
                for qi, q in enumerate(q_values):
                    row = {'Method': method, 'Noise_pct': np_p,
                           'Outlier_pct': op, 'Q': q}
                    for si, sn in enumerate(stat_names):
                        row[f'Jaccard_{sn}'] = round(J_stats[k, ni, oi, qi, si], 4)
                    flat_rows.append(row)
    path = os.path.join(OUTPUT_DIR, f'B2_Jaccard_Full_{DATE}.csv')
    pd.DataFrame(flat_rows).to_csv(path, index=False); print(f"  Saved: {path}")

    # ── Index helpers ─────────────────────────────────────────────────────────
    ni0 = noise_pct.index(0)
    oi0 = outlier_pct.index(0)
    qi0 = q_values.index(0)
    q_labels = [str(q) for q in q_values]
    o_labels = [f'{op}%' for op in outlier_pct]
    n_labels = [f'{np_p}%' for np_p in noise_pct]

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B2-1 — Three-panel line plots (noise=0%): vs Q, vs Outliers, vs Noise
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Jaccard vs quantization q (outlier=0%, noise=0%)
    ax = axes[0]
    for k, method in enumerate(METHODS):
        ax.plot(range(nQ), J_mean[k, ni0, oi0, :],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
        ax.fill_between(range(nQ),
                        np.clip(J_mean[k, ni0, oi0, :] - J_std[k, ni0, oi0, :], 0, 1),
                        np.clip(J_mean[k, ni0, oi0, :] + J_std[k, ni0, oi0, :], 0, 1),
                        color=COLORS[method], alpha=0.08)
    ax.set_xticks(range(nQ)); ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_xlabel('Quantization Step q')
    ax.set_ylabel('Mean Jaccard Index ±1σ')
    ax.set_title('(a) vs Spatial Quantization\n(noise=0%, outliers=0%)')
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)

    # Panel B: Jaccard vs outlier fraction (q=0, noise=0%)
    ax = axes[1]
    for k, method in enumerate(METHODS):
        ax.plot(outlier_pct, J_mean[k, ni0, :, qi0],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
        ax.fill_between(outlier_pct,
                        np.clip(J_mean[k, ni0, :, qi0] - J_std[k, ni0, :, qi0], 0, 1),
                        np.clip(J_mean[k, ni0, :, qi0] + J_std[k, ni0, :, qi0], 0, 1),
                        color=COLORS[method], alpha=0.08)
    ax.set_xlabel('Outlier Fraction (%)')
    ax.set_title('(b) vs Outlier Fraction\n(noise=0%, q=0 continuous)')
    ax.set_ylim(0, 1.05)

    # Panel C: Jaccard vs noise level (q=0, outliers=0%)
    ax = axes[2]
    for k, method in enumerate(METHODS):
        ax.plot(noise_pct, J_mean[k, :, oi0, qi0],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
        ax.fill_between(noise_pct,
                        np.clip(J_mean[k, :, oi0, qi0] - J_std[k, :, oi0, qi0], 0, 1),
                        np.clip(J_mean[k, :, oi0, qi0] + J_std[k, :, oi0, qi0], 0, 1),
                        color=COLORS[method], alpha=0.08)
    ax.set_xlabel('Noise Level (% of radius)')
    ax.set_title('(c) vs Noise Level\n(outliers=0%, q=0 continuous)')
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)

    fig.suptitle('Experiment B2 — Full Circle Performance Analysis\n'
                 f'Center=({B2_X0},{B2_Y0}), r={B2_R0}, N={B2_N_POINTS}, {N_ITER_B} trials',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B2_Fig1_Panel_Lines_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B2-2 — Heatmap: best method per (outlier%, q) at noise=0%
    # ══════════════════════════════════════════════════════════════════════════
    best_n0 = best[ni0]   # (nO, nQ)
    cmap_disc = plt.cm.get_cmap('tab10', n_meth)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(best_n0, aspect='auto', cmap=cmap_disc, vmin=0, vmax=n_meth - 1)
    ax.set_xticks(range(nQ)); ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_yticks(range(nO)); ax.set_yticklabels(o_labels, fontsize=10)
    ax.set_xlabel('Quantization Step q (coarser →)')
    ax.set_ylabel('Outlier Fraction p')
    ax.set_title('Experiment B2 — Best Method per Configuration (noise=0%)',
                 pad=8)
    for oi2 in range(nO):
        for qi2 in range(nQ):
            m_idx = best_n0[oi2, qi2]
            jac   = J_mean[m_idx, ni0, oi2, qi2]
            col   = 'white' if cmap_disc(m_idx)[0] < 0.5 else 'black'
            ax.text(qi2, oi2,
                    f'{METHODS[m_idx][:6]}\n{jac:.3f}',
                    ha='center', va='center', fontsize=6.5,
                    color=col, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_meth), fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(METHODS, fontsize=9)
    cbar.set_label('Method', fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B2_Fig2_Heatmap_BestMethod_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B2-3 — Jaccard heatmaps at 3 noise levels (noise=0%, 2%, 5%)
    # ══════════════════════════════════════════════════════════════════════════
    noise_panel = [0, 2, 5]
    ni_panel    = [noise_pct.index(n) for n in noise_panel if n in noise_pct]

    fig, axes = plt.subplots(1, len(ni_panel),
                              figsize=(7 * len(ni_panel), 6),
                              sharey=True)
    cmap_j = LinearSegmentedColormap.from_list('jac2', ['#d73027','#fee090','#4575b4'], N=256)
    for ax, ni in zip(axes, ni_panel):
        best_ni = np.argmax(J_mean[:, ni, :, :], axis=0)  # (nO, nQ)
        data    = J_mean[best_ni, ni, np.arange(nO)[:, None],
                         np.arange(nQ)[None, :]]           # max-J per cell
        im2 = ax.imshow(data, aspect='auto', cmap=cmap_j, vmin=0, vmax=1)
        ax.set_xticks(range(nQ)); ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_yticks(range(nO)); ax.set_yticklabels(o_labels, fontsize=9)
        ax.set_title(f'Noise = {noise_pct[ni]}% of r₀', fontsize=11)
        ax.set_xlabel('Quantization q')
        for oi2 in range(nO):
            for qi2 in range(nQ):
                ax.text(qi2, oi2, f'{data[oi2, qi2]:.2f}',
                        ha='center', va='center', fontsize=6, color='white')
        ax.grid(False)
    axes[0].set_ylabel('Outlier Fraction p')
    cbar2 = plt.colorbar(im2, ax=axes[-1], fraction=0.03, pad=0.02)
    cbar2.set_label('Best Jaccard (over methods)', fontsize=10)
    fig.suptitle('Experiment B2 — Best Achievable Jaccard per Configuration',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B2_Fig3_Heatmap_NoisePanels_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE B2-4 — Win count bar chart
    # ══════════════════════════════════════════════════════════════════════════
    wins = [int(np.sum(best == k)) for k in range(n_meth)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(METHODS, wins,
                  color=[COLORS[m] for m in METHODS],
                  edgecolor='white', linewidth=0.5, width=0.7)
    for bar, w in zip(bars, wins):
        pct = 100 * w / total_cells
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_cells * 0.005,
                f'{w}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Configurations Won')
    ax.set_title(f'Experiment B2 — Win Count (best Jaccard per config)\n'
                 f'Total: {total_cells} configurations '
                 f'({nN} noise × {nO} outlier × {nQ} quantization levels)')
    ax.set_ylim(0, max(wins) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'B2_Fig4_WinCount_{DATE}.png')
    plt.savefig(path); plt.savefig(path.replace('.png', '.pdf')); plt.close()
    print(f"  Saved: {path}")


def print_summary_B2(res):
    print("\n" + "=" * 70)
    print("Experiment B2 — Win Counts (best Jaccard per noise×outlier×q cell)")
    print("=" * 70)
    J_mean      = res['J_stats'][..., 1]
    total       = J_mean.shape[1] * J_mean.shape[2] * J_mean.shape[3]
    best        = np.argmax(J_mean, axis=0)
    print(f"  Total cells: {total}")
    for k, method in enumerate(METHODS):
        wins = int(np.sum(best == k))
        mark = ' ★' if method == '3C-FBI' else ''
        print(f"  {method:<10}: {wins:4d} / {total}  ({100*wins/total:.1f}%){mark}")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("3C-FBI: Comprehensive Circle Fitting Evaluation")
    print("Exp A methods: CIBICA, 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi")
    print("Exp B methods: 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi")
    print("=" * 70)

    # ── Experiment A ──────────────────────────────────────────────────────────
    print("\n>>> EXPERIMENT A: Real-world Parkinson's disease data")
    res_A = run_experiment_A()
    print("\nSaving Experiment A outputs...")
    save_experiment_A(res_A)
    print_summary_A(res_A)

    # ── Experiment B1 ─────────────────────────────────────────────────────────
    print("\n>>> EXPERIMENT B1: Synthetic semicircle with varying outliers")
    res_B1 = run_experiment_B1()
    print("\nSaving Experiment B1 outputs...")
    save_experiment_B1(res_B1)
    print_summary_B1(res_B1)

    # ── Experiment B2 ─────────────────────────────────────────────────────────
    print("\n>>> EXPERIMENT B2: Full circle — noise × outliers × quantization")
    res_B2 = run_experiment_B2()
    print("\nSaving Experiment B2 outputs...")
    save_experiment_B2(res_B2)
    print_summary_B2(res_B2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
