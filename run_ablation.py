"""
run_ablation.py — CIBICA ablation study (safe variants only)

Evaluates three variants of CIBICA on real clinical frames × 18 preprocessing
configs.  CIBICA.py is NOT modified.

Variants
--------
  full           — complete CIBICA (vectorized_XYR → median_3d → LS refinement)
  no_refinement  — skip LS step (uses CIBICA built-in refinement=False flag)
  no_consensus   — replace median_3d with np.median on each dimension separately;
                   i.e. mode-based consensus is replaced by independent-axis median

Why no_constraints is excluded
-------------------------------
The geometric constraint filtering (bounds + radius size checks) inside
vectorized_XYR is structurally coupled to the radius computation: p1 is
progressively deleted in synchrony with cx and cy across four sequential
filtering passes, and radius is then computed from the *filtered* p1
(line 74: radius = sqrt((cx - p1[:,0])**2 + (cy - p1[:,1])**2)).
Bypassing the filters from outside the function would leave p1 with
all rows while cx/cy are reduced, producing a shape mismatch or
incorrect radius values.  A correct no_constraints variant requires
modifying vectorized_XYR itself, which we deliberately avoid.

Usage
-----
    cd /Users/erc/Documents/3C-FBI-Circle-fitting
    conda activate poseestimation
    python run_ablation.py
"""

import math as m
import os
import time
from datetime import date

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Import CIBICA internals (no changes to CIBICA.py)
from algorithms.CIBICA import CIBICA, vectorized_XYR, LS_circle
from algorithms.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)
from scipy.spatial.distance import cdist
import random
from itertools import combinations

# ============================================================================
# Global plot settings
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
})

# ============================================================================
# Constants
# ============================================================================

DATE    = date.today().strftime('%Y%m%d')
OUTPUT  = 'ablation_results'
os.makedirs(OUTPUT, exist_ok=True)

VARIANTS = ['full', 'no_refinement', 'no_consensus']

COLORS = {
    'full':          '#1f77b4',   # blue
    'no_refinement': '#d62728',   # red
    'no_consensus':  '#2ca02c',   # green
}

MARKERS = {
    'full':          'o',
    'no_refinement': 's',
    'no_consensus':  '^',
}

LINESTYLES = {
    'full':          'solid',
    'no_refinement': 'dashed',
    'no_consensus':  (0, (5, 2)),
}

LABELS = {
    'full':          'CIBICA (full)',
    'no_refinement': 'CIBICA – no refinement',
    'no_consensus':  'CIBICA – no consensus',
}

BEST_GL = ['GL80', 'GL82', 'GL84']

N_TRIPLETS = 500

# ============================================================================
# CIBICA variant implementations
# ============================================================================

def cibica_full(coord, xmax=50, ymax=50):
    """Full CIBICA: uses built-in refinement=True (default)."""
    return CIBICA(coord, n_triplets=N_TRIPLETS, xmax=xmax, ymax=ymax, refinement=True)


def cibica_no_refinement(coord, xmax=50, ymax=50):
    """CIBICA without LS refinement step: uses built-in refinement=False."""
    return CIBICA(coord, n_triplets=N_TRIPLETS, xmax=xmax, ymax=ymax, refinement=False)


def cibica_no_consensus(coord, xmax=50, ymax=50):
    """
    CIBICA without mode-based consensus.

    Replaces median_3d (mode on encoded identifier) with independent per-axis
    np.median on cx, cy, radius.  Refinement step is kept identical to full.

    Note: coordinate return order matches CIBICA convention: (col, row, radius).
    """
    if len(coord) < 3:
        return np.nan, np.nan, np.nan

    combi = list(combinations(np.arange(len(coord)), 3))
    N = min(N_TRIPLETS, len(combi))
    rs = np.array(random.sample(combi, N))

    p1 = coord[rs[:, 0]]
    p2 = coord[rs[:, 1]]
    p3 = coord[rs[:, 2]]

    cx_arr, cy_arr, r_arr = vectorized_XYR(p1, p2, p3, xmax, ymax)

    if len(cx_arr) == 0:
        return np.nan, np.nan, np.nan

    # Replace mode-based consensus with per-axis median
    cx_med = float(np.median(cx_arr))
    cy_med = float(np.median(cy_arr))
    r_med  = float(np.median(r_arr))

    # Least-squares refinement (same as full variant)
    coord2    = [(cx_med, cy_med)]
    near      = np.where(np.abs(cdist(coord2, coord) - r_med) < 1.5)
    circle_pts = coord[near[1]]

    if len(circle_pts) >= 3:
        xl, yl, rl, _ = np.round(LS_circle(circle_pts[:, 0], circle_pts[:, 1]), 3)
        return float(yl), float(xl), float(rl)
    else:
        return cy_med, cx_med, r_med


# ============================================================================
# Utilities
# ============================================================================

def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Analytical Jaccard index (IoU) for two circles."""
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d == 0:
        return float(min((r1/r2)**2, (r2/r1)**2))
    R, r = max(r1, r2), min(r1, r2)
    if d >= r1 + r2:
        return 0.0
    if d <= R - r:
        return float((r/R)**2)
    d1 = (d**2 + r1**2 - r2**2) / (2*d)
    d2 = d - d1
    a1 = 2 * m.acos(max(-1.0, min(1.0, d1/r1)))
    a2 = 2 * m.acos(max(-1.0, min(1.0, d2/r2)))
    inter = 0.5*r1**2*(a1 - m.sin(a1)) + 0.5*r2**2*(a2 - m.sin(a2))
    union = m.pi*(R**2 + r**2) - inter
    return float(inter / union)


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _hl_estimator_ci(a, b, n_boot=4000, ci_level=0.95, seed=42):
    """Hodges-Lehmann estimator and bootstrap 95% CI for paired differences a − b."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    n    = len(diff)
    ii, jj = np.triu_indices(n, k=0)
    hl   = float(np.median((diff[ii] + diff[jj]) / 2.0))
    rng  = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for s in range(n_boot):
        samp    = rng.choice(diff, size=n, replace=True)
        wi, wj  = np.triu_indices(n, k=0)
        boot[s] = np.median((samp[wi] + samp[wj]) / 2.0)
    alpha = 1.0 - ci_level
    return hl, float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


# ============================================================================
# Experiment runner
# ============================================================================

VARIANT_FN = {
    'full':          cibica_full,
    'no_refinement': cibica_no_refinement,
    'no_consensus':  cibica_no_consensus,
}


def run_ablation():
    """
    Run all three CIBICA variants on all images × 18 preprocessing configs.
    Returns dict with Jaccard[variant] and Time_s[variant] arrays.
    """
    ground_truth = pd.read_csv('data/Ground_Truth.csv')
    filenames    = ground_truth['Filename'].tolist()
    configs      = get_preprocessing_configs()

    n_images  = len(filenames)
    n_configs = len(configs)

    Jaccard = {v: np.zeros((n_images, n_configs)) for v in VARIANTS}
    Time_s  = {v: np.zeros((n_images, n_configs)) for v in VARIANTS}

    print(f"Ablation study: {n_images} images × {n_configs} preprocessing configs")
    print(f"Variants: {', '.join(VARIANTS)}")
    print("=" * 70)
    t_start = time.time()

    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']

        BS_crop = cv2.imread(os.path.join('data', 'black_sphere_ROI', filename + '.png'))
        G_crop  = cv2.imread(os.path.join('data', 'green_back_ROI',   filename + '.png'))
        if BS_crop is None:
            print(f"  Warning: missing {filename} — skipping")
            continue

        xmax = BS_crop.shape[1]
        ymax = BS_crop.shape[0]

        for j, cfg in enumerate(configs):
            try:
                if cfg['green_level'] is not None:
                    _, edge_img, edgels = preprocess_green_level(BS_crop, cfg['green_level'])
                else:
                    _, edge_img, edgels = preprocess_median_filter(BS_crop, G_crop, cfg['median_size'])
            except Exception:
                continue

            if len(edgels) < 3:
                continue

            for variant, fn in VARIANT_FN.items():
                t0 = time.perf_counter()
                try:
                    x_c, y_c, r_c = fn(edgels, xmax=xmax, ymax=ymax)
                    Time_s[variant][i, j] = time.perf_counter() - t0
                    if not (np.isnan(x_c) or r_c is None or r_c <= 0):
                        # CIBICA returns (col, row, r) — same convention as jaccard call in main
                        Jaccard[variant][i, j] = jaccard_circles(XGT, YGT, RGT, x_c, y_c, r_c)
                except Exception:
                    Time_s[variant][i, j] = time.perf_counter() - t0

        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            print(f"  {i+1}/{n_images} images  ({time.time()-t_start:.1f}s)")

    print("=" * 70)
    print(f"Done in {time.time()-t_start:.1f}s")

    return {
        **{f'Jaccard_{v}': Jaccard[v] for v in VARIANTS},
        **{f'Time_{v}':    Time_s[v]  for v in VARIANTS},
        'config_names': [c['name'] for c in configs],
        'filenames':    filenames,
    }


# ============================================================================
# Save CSVs
# ============================================================================

def save_csvs(results):
    cfg_names = results['config_names']
    fnames    = results['filenames']

    for v in VARIANTS:
        df = pd.DataFrame(
            results[f'Jaccard_{v}'],
            index=fnames,
            columns=cfg_names,
        )
        df.to_csv(os.path.join(OUTPUT, f'Ablation_Jaccard_{v}_{DATE}.csv'))

    # Summary: mean Jaccard per config across all images
    rows = []
    for v in VARIANTS:
        J = results[f'Jaccard_{v}']
        row = {'Variant': v}
        for j, name in enumerate(cfg_names):
            row[name] = round(float(J[:, j].mean()), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT, f'Ablation_Summary_{DATE}.csv'), index=False
    )

    # Statistical comparison: full vs each ablated variant
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    scores   = {v: results[f'Jaccard_{v}'][:, best_idx].mean(axis=1) for v in VARIANTS}

    stat_rows = []
    for v in ['no_refinement', 'no_consensus']:
        diff = scores['full'] - scores[v]
        if np.all(diff == 0):
            stat_rows.append({'Variant': v, 'HL': 0, 'CI_lo': 0, 'CI_hi': 0,
                               'p_value': np.nan, 'Stars': 'ns'})
            continue
        stat, p = wilcoxon(scores['full'], scores[v], alternative='two-sided')
        hl, lo, hi = _hl_estimator_ci(scores['full'], scores[v])
        stat_rows.append({
            'Variant': v,
            'HL':      round(hl, 4),
            'CI_lo':   round(lo, 4),
            'CI_hi':   round(hi, 4),
            'W_stat':  round(float(stat), 1),
            'p_value': float(p),
            'Stars':   _sig_stars(p),
        })
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(OUTPUT, f'Ablation_Stats_{DATE}.csv'), index=False
    )
    print(f"CSVs saved to {OUTPUT}/")


# ============================================================================
# Plots
# ============================================================================

def save_plots(results):
    cfg_names = results['config_names']
    n_cfg     = len(cfg_names)

    # ── Fig 1: Mean Jaccard per preprocessing config ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(n_cfg)
    for v in VARIANTS:
        J    = results[f'Jaccard_{v}']
        mean = J.mean(axis=0)
        ax.plot(x, mean,
                color=COLORS[v], marker=MARKERS[v],
                linestyle=LINESTYLES[v], label=LABELS[v],
                linewidth=2.0, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Preprocessing configuration')
    ax.set_ylabel('Mean Jaccard index')
    ax.set_title('CIBICA ablation — Jaccard per preprocessing config')
    ax.legend(loc='lower right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT, f'Ablation_Fig1_Line_{DATE}.{ext}'))
    plt.close(fig)

    # ── Fig 2: Violin plots (best 3 GL configs) ────────────────────────────────
    best_idx = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    scores   = {v: results[f'Jaccard_{v}'][:, best_idx].mean(axis=1) for v in VARIANTS}

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(VARIANTS))
    parts = ax.violinplot(
        [scores[v] for v in VARIANTS],
        positions=positions,
        widths=0.6, showmedians=True, showextrema=True,
    )
    for pc, v in zip(parts['bodies'], VARIANTS):
        pc.set_facecolor(COLORS[v])
        pc.set_alpha(0.55)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    rng = np.random.default_rng(0)
    for k, v in enumerate(VARIANTS):
        jitter = rng.uniform(-0.1, 0.1, len(scores[v]))
        ax.scatter(positions[k] + jitter, scores[v],
                   color=COLORS[v], s=12, alpha=0.55, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS], rotation=15, ha='right')
    ax.set_ylabel('Jaccard index (mean over GL80–GL84)')
    ax.set_title('CIBICA ablation — distribution comparison')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT, f'Ablation_Fig2_Violin_{DATE}.{ext}'))
    plt.close(fig)

    # ── Fig 3: Lollipop — HL ± 95% CI (full vs ablated) ──────────────────────
    comparisons = ['no_refinement', 'no_consensus']
    hls, los, his, stars_list = [], [], [], []
    for v in comparisons:
        diff = scores['full'] - scores[v]
        if np.all(diff == 0):
            hls.append(0.); los.append(0.); his.append(0.); stars_list.append('ns')
            continue
        _, p = wilcoxon(scores['full'], scores[v], alternative='two-sided')
        hl, lo, hi = _hl_estimator_ci(scores['full'], scores[v])
        hls.append(hl); los.append(lo); his.append(hi)
        stars_list.append(_sig_stars(p))

    fig, ax = plt.subplots(figsize=(6, 4))
    ypos = np.arange(len(comparisons))
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    for k, v in enumerate(comparisons):
        ax.hlines(ypos[k], los[k], his[k], color=COLORS[v], linewidth=4, alpha=0.5)
        ax.plot(hls[k], ypos[k], 'o', color=COLORS[v], markersize=9, zorder=5)
        ax.text(his[k] + 0.005, ypos[k], stars_list[k],
                va='center', ha='left', fontsize=12, color=COLORS[v])

    ax.set_yticks(ypos)
    ax.set_yticklabels([f'full vs {LABELS[v]}' for v in comparisons])
    ax.set_xlabel('Hodges-Lehmann Δ Jaccard (full − ablated)  ±95% CI')
    ax.set_title('Ablation: contribution of each component')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT, f'Ablation_Fig3_Lollipop_{DATE}.{ext}'))
    plt.close(fig)

    # ── Fig 4: Mean FPS per variant ────────────────────────────────────────────
    mean_fps = {}
    for v in VARIANTS:
        T = results[f'Time_{v}']
        T_pos = T[T > 0]
        mean_fps[v] = 1.0 / T_pos.mean() if len(T_pos) else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [LABELS[v] for v in VARIANTS],
        [mean_fps[v] for v in VARIANTS],
        color=[COLORS[v] for v in VARIANTS],
        edgecolor='black', linewidth=0.6, alpha=0.85,
    )
    for bar, v in zip(bars, VARIANTS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean_fps[v]:.1f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Mean FPS')
    ax.set_title('Ablation — processing speed (FPS)')
    plt.xticks(rotation=15, ha='right')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUTPUT, f'Ablation_Fig4_FPS_{DATE}.{ext}'))
    plt.close(fig)

    print(f"Figures saved to {OUTPUT}/")


# ============================================================================
# Print summary
# ============================================================================

def print_summary(results):
    cfg_names = results['config_names']
    best_idx  = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    scores    = {v: results[f'Jaccard_{v}'][:, best_idx].mean(axis=1) for v in VARIANTS}

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY  (mean Jaccard on GL80/GL82/GL84)")
    print("=" * 60)
    for v in VARIANTS:
        s = scores[v]
        print(f"  {LABELS[v]:<35s}  mean={s.mean():.4f}  median={np.median(s):.4f}  std={s.std():.4f}")

    print("\nWilcoxon tests (full vs ablated):")
    for v in ['no_refinement', 'no_consensus']:
        diff = scores['full'] - scores[v]
        if np.all(diff == 0):
            print(f"  full vs {v}: no difference")
            continue
        stat, p = wilcoxon(scores['full'], scores[v], alternative='two-sided')
        hl, lo, hi = _hl_estimator_ci(scores['full'], scores[v])
        print(f"  full vs {LABELS[v]}:")
        print(f"    HL={hl:+.4f}  95%CI=[{lo:.4f}, {hi:.4f}]  p={p:.4e} {_sig_stars(p)}")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    results = run_ablation()
    save_csvs(results)
    save_plots(results)
    print_summary(results)
