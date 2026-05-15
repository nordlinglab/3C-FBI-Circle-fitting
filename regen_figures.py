"""
regen_figures.py — Regenerate all publication figures from saved 20260501 CSVs.

Reads the CSV outputs produced by main_3C_FBI.py (20260501 run) and reproduces
every figure without re-running the 8–14 h experiments.

Usage:
    conda activate poseestimation
    python regen_figures.py [--date YYYYMMDD]

By default uses the most-recent available CSV date found in CCC_FBI_results/.
Pass --date 20260501 to pin to that run explicitly.
"""

import argparse
import os
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd

# ============================================================================
# Style — must match main_3C_FBI.py exactly
# ============================================================================
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.labelsize':     12,
    'axes.labelweight':   'bold',
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9.5,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.8',
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'lines.linewidth':    2.0,
    'lines.markersize':   6,
    'errorbar.capsize':   3,
})

METHODS_A       = ['CIBICA','3C-FBI','RHT','RCD','RFCA','Nurunnabi','Guo','Greco','Qi']
METHODS_A_PAPER = ['3C-FBI','RHT','RCD','RFCA','Nurunnabi','Guo','Greco','Qi']
METHODS_B       = ['CIBICA','3C-FBI','RHT','RCD','RFCA','Nurunnabi','Guo','Greco','Qi']
BEST_GL         = ['GL80','GL82','GL84']

COLORS = {
    'CIBICA':    '#2ca02c',
    '3C-FBI':    '#1f77b4',
    'RHT':       '#d62728',
    'RCD':       '#ff7f0e',
    'RFCA':      '#9467bd',
    'Nurunnabi': '#8c564b',
    'Guo':       '#e377c2',
    'Greco':     '#7f7f7f',
    'Qi':        '#bcbd22',
}
MARKERS = {
    'CIBICA':    's',
    '3C-FBI':    'o',
    'RHT':       '^',
    'RCD':       'v',
    'RFCA':      'D',
    'Nurunnabi': 'P',
    'Guo':       'X',
    'Greco':     'h',
    'Qi':        '*',
}
LINESTYLES = {
    'CIBICA':    (0,(3,1,1,1)),
    '3C-FBI':    'solid',
    'RHT':       'dashed',
    'RCD':       'dotted',
    'RFCA':      (0,(5,2)),
    'Nurunnabi': (0,(3,2)),
    'Guo':       (0,(1,1)),
    'Greco':     (0,(5,1,1,1)),
    'Qi':        (0,(4,2,1,2)),
}

OUTPUT_DIR = 'CCC_FBI_results'
B1_X0, B1_Y0, B1_R0 = 50,  60,  100
B2_X0, B2_Y0, B2_R0 = 120, 120, 120
B2_N_POINTS          = 100


# ============================================================================
# Helpers
# ============================================================================

def _savefig(path):
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f"  Saved: {path}")


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _plot_violin_strip(data_list, method_list, title, ylabel, path, tick_labels=None):
    n = len(method_list)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 5.5))
    for i, (vals, method) in enumerate(zip(data_list, method_list)):
        v = np.asarray(vals)
        vp = ax.violinplot(v, positions=[i], widths=0.7, showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(COLORS[method]); body.set_alpha(0.35)
            body.set_edgecolor(COLORS[method])
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q25, q75, color=COLORS[method], linewidth=5, alpha=0.6)
        ax.scatter(i, med, color='white', s=45, zorder=5,
                   edgecolors=COLORS[method], linewidth=1.5)
        jitter = np.random.default_rng(42+i).uniform(-0.15, 0.15, len(v))
        ax.scatter(i + jitter, v, color=COLORS[method], alpha=0.25, s=12, zorder=3, linewidths=0)
        ax.scatter(i, np.mean(v), marker='D', color=COLORS[method],
                   s=40, zorder=6, edgecolors='white', linewidth=0.8)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tick_labels if tick_labels else method_list,
                       fontsize=9, rotation=15 if tick_labels else 0, ha='right')
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(max(0, min(np.concatenate(data_list)) - 0.05), 1.02)
    ax.set_ylabel(ylabel); ax.set_title(title)
    patches = [mpatches.Patch(facecolor=COLORS[m], alpha=0.6, label=m) for m in method_list]
    ax.legend(handles=patches, loc='lower right', ncol=3, fontsize=8.5)
    plt.tight_layout()
    _savefig(path)


def _plot_focal_stats(df_fs, path):
    methods = df_fs['Baseline'].tolist()
    hls     = df_fs['HL'].tolist()
    lo      = df_fs['CI_lo'].tolist()
    hi      = df_fs['CI_hi'].tolist()
    stars   = df_fs['Stars'].tolist()
    r_rbs   = df_fs['r_rb'].tolist()
    n = len(methods)
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(n)
    for i in range(n):
        col = '#2166ac' if hls[i] > 0 else '#d6604d'
        ax.plot([lo[i], hi[i]], [i, i], color=col, linewidth=2.5, solid_capstyle='round', zorder=2)
        ax.scatter(hls[i], i, color=col, s=90, zorder=4, edgecolors='white', linewidth=1.0)
        x_txt = hi[i] + abs(max(hi) - min(lo)) * 0.03
        ax.text(x_txt, i, f'{stars[i]}  |r_rb|={abs(r_rbs[i]):.2f}',
                va='center', fontsize=9, color=col, fontweight='bold')
    ax.axvline(0, color='0.3', linewidth=1.2, linestyle='--', alpha=0.7, label='No difference')
    ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Hodges-Lehmann Δ Jaccard (3C-FBI − Baseline)\nwith 95% bootstrap CI')
    ax.set_title('Statistical Comparison: 3C-FBI vs Baseline Methods\n'
                 '(Wilcoxon signed-rank, GL80/GL82/GL84, n=144 frames)', pad=8)
    ax.legend(fontsize=9, loc='lower right')
    xlims = ax.get_xlim()
    ax.axvspan(0, xlims[1], alpha=0.04, color='#2166ac')
    ax.axvspan(xlims[0], 0, alpha=0.04, color='#d6604d')
    ax.set_xlim(xlims)
    plt.tight_layout()
    _savefig(path)


def generate_semicircle_points(x0=50, y0=60, r0=100, n_points=50,
                               noise_std=1.0, n_outliers=0, rng=None):
    if rng is None: rng = np.random.default_rng()
    theta = rng.uniform(0, np.pi, n_points)
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
    if rng is None: rng = np.random.default_rng()
    theta  = rng.uniform(0, 2 * np.pi, n_points)
    r_noise = rng.normal(0, noise_std, n_points)
    x = x0 + (r0 + r_noise) * np.cos(theta)
    y = y0 + (r0 + r_noise) * np.sin(theta)
    return np.column_stack([x, y])


def apply_quantization(points, q):
    if q == 0: return points
    return np.unique(np.round(points / q), axis=0)


# ============================================================================
# Data loading
# ============================================================================

def find_csv_date(prefix, result_dir=OUTPUT_DIR):
    """Find latest available date tag for a given CSV prefix."""
    pattern = os.path.join(result_dir, f'{prefix}_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV matching {pattern}")
    latest = files[-1]
    tag = os.path.basename(latest).replace(f'{prefix}_', '').replace('.csv', '')
    return tag


def load_exp_A(date_tag):
    """Load per-frame Jaccard + timing + stats CSVs for Experiment A."""
    base = OUTPUT_DIR
    cfg_names = ['GL70','GL72','GL74','GL76','GL78','GL80','GL82','GL84','GL86',
                 'Med3','Med5','Med7','Med9','Med11','Med13','Med15','Med17','Med19']

    # Per-method per-frame Jaccard: shape (n_cfg, n_frames) per method
    jac = {}
    filenames = None
    for m in METHODS_A:
        path = os.path.join(base, f'A_Jaccard_{m}_{date_tag}.csv')
        df = pd.read_csv(path, index_col=0)
        jac[m] = df[cfg_names].values.T   # (n_cfg, n_frames)
        if filenames is None:
            filenames = df.index.tolist()

    # Timing: per method per config mean time
    tim_df = pd.read_csv(os.path.join(base, f'A_Timing_Raw_{date_tag}.csv'))
    tim = {}  # method → array (n_cfg,) of mean times
    for m in METHODS_A:
        sub = tim_df[tim_df['Method'] == m].set_index('Config')
        tim[m] = np.array([sub.loc[c, 'Time_s'] if c in sub.index else np.nan
                           for c in cfg_names])

    # Focal stats (pre-computed)
    df_focal_A  = pd.read_csv(os.path.join(base, f'A_Stats_FocalTest_{date_tag}.csv'))
    df_focal_Ap = pd.read_csv(os.path.join(base, f'Ap_Stats_FocalTest_{date_tag}.csv'))

    # Pairwise stats
    df_pair_A  = pd.read_csv(os.path.join(base, f'A_Stats_Pairwise_{date_tag}.csv'))
    df_pair_Ap = pd.read_csv(os.path.join(base, f'Ap_Stats_Pairwise_{date_tag}.csv'))

    # Summary tables (AD, RMSE, FPS pre-aggregated)
    df_gl_A  = pd.read_csv(os.path.join(base, f'A_Table1_Best3GL_{date_tag}.csv'), index_col=0)
    df_gl_Ap = pd.read_csv(os.path.join(base, f'Ap_Table1_Best3GL_{date_tag}.csv'), index_col=0)

    return dict(jac=jac, tim=tim, cfg_names=cfg_names, filenames=filenames,
                focal_A=df_focal_A, focal_Ap=df_focal_Ap,
                pair_A=df_pair_A, pair_Ap=df_pair_Ap,
                gl_A=df_gl_A, gl_Ap=df_gl_Ap)


def load_exp_B1(date_tag):
    base = OUTPUT_DIR
    j   = pd.read_csv(os.path.join(base, f'B1_Jaccard_{date_tag}.csv'), index_col=0)
    ad  = pd.read_csv(os.path.join(base, f'B1_AD_mm_{date_tag}.csv'),   index_col=0)
    rmse= pd.read_csv(os.path.join(base, f'B1_RMSE_mm_{date_tag}.csv'), index_col=0)
    tim = pd.read_csv(os.path.join(base, f'B1_Timing_{date_tag}.csv'),  index_col=0)
    # outlier counts from column names like "0 outliers"
    out_cols = [c for c in j.columns if 'outliers' in c]
    outs = [int(c.split()[0]) for c in out_cols]
    return dict(J=j, AD=ad, RMSE=rmse, Tim=tim, outs=outs)


def load_exp_B2(date_tag):
    base = OUTPUT_DIR
    df   = pd.read_csv(os.path.join(base, f'B2_Jaccard_Full_{date_tag}.csv'))
    wins = pd.read_csv(os.path.join(base, f'B2_Table3_WinCount_{date_tag}.csv'), index_col=0)
    tim  = pd.read_csv(os.path.join(base, f'B2_Timing_{date_tag}.csv'), index_col=0)

    noise_pct   = sorted(df['Noise_pct'].unique().tolist())
    outlier_pct = sorted(df['Outlier_pct'].unique().tolist())
    q_values    = sorted(df['Q'].unique().tolist())
    nN, nO, nQ  = len(noise_pct), len(outlier_pct), len(q_values)
    n_meth      = len(METHODS_B)

    J_mean = np.zeros((n_meth, nN, nO, nQ))
    for k, m in enumerate(METHODS_B):
        sub = df[df['Method'] == m]
        for ni, np_v in enumerate(noise_pct):
            for oi, op in enumerate(outlier_pct):
                for qi, q in enumerate(q_values):
                    row = sub[(sub['Noise_pct'] == np_v) &
                              (sub['Outlier_pct'] == op) &
                              (sub['Q'] == q)]
                    if len(row):
                        J_mean[k, ni, oi, qi] = row['Jaccard_Mean'].values[0]

    return dict(J_mean=J_mean, wins=wins, tim=tim,
                noise_pct=noise_pct, outlier_pct=outlier_pct, q_values=q_values)


# ============================================================================
# Experiment A figures
# ============================================================================

def save_A_figures(d, methods, tag, out_date):
    """Generate all Exp A figures for one method set (tag='A' or 'Ap')."""
    jac       = d['jac']
    tim       = d['tim']
    cfg_names = d['cfg_names']
    n_cfg     = len(cfg_names)
    n_meth    = len(methods)
    best_idx  = [cfg_names.index(gl) for gl in BEST_GL if gl in cfg_names]
    df_gl     = d[f'gl_{tag}']
    df_focal  = d[f'focal_{tag}']
    df_pair   = d[f'pair_{tag}']

    # ── Fig 1: line plot all 18 configs ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(n_cfg)
    for method in methods:
        mean_j = jac[method].mean(axis=1)
        ax.plot(x, mean_j, color=COLORS[method], linewidth=2.0,
                marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], label=method, zorder=3)
    ax.axvspan(-0.5, 8.5, alpha=0.04, color='steelblue')
    ax.axvspan(8.5, n_cfg - 0.5, alpha=0.04, color='darkorange')
    ax.axvline(x=8.5, color='0.5', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(4,    0.55, 'Green-Level',  ha='center', fontsize=9, color='steelblue',  alpha=0.8)
    ax.text(13.5, 0.55, 'Median Filter',ha='center', fontsize=9, color='darkorange', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0.5, 1.02); ax.set_xlim(-0.5, n_cfg - 0.5)
    ax.set_xlabel('Preprocessing Configuration'); ax.set_ylabel('Mean Jaccard Index')
    ax.set_title(f'Experiment A — Mean Jaccard Index across 18 Preprocessing Configurations\n'
                 f'(144 clinical frames, {n_meth} methods)')
    ax.legend(loc='lower left', ncol=3, fontsize=9)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig1_Jaccard_AllConfigs_{out_date}.png'))

    # ── Fig 1b: dual-panel GL | Med ──────────────────────────────────────────
    gl_idx  = [i for i, c in enumerate(cfg_names) if c.startswith('GL')]
    med_idx = [i for i, c in enumerate(cfg_names) if c.startswith('Med')]
    gl_names  = [cfg_names[i] for i in gl_idx]
    med_names = [cfg_names[i] for i in med_idx]
    fig, (ax_gl, ax_med) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, idx_list, names, panel_title, bg_color in [
        (ax_gl,  gl_idx,  gl_names,  'Green-Level preprocessing',   'steelblue'),
        (ax_med, med_idx, med_names, 'Median Filter preprocessing',  'darkorange'),
    ]:
        x2 = np.arange(len(idx_list))
        for method in methods:
            mean_j = jac[method][idx_list, :].mean(axis=1)
            ax.plot(x2, mean_j, color=COLORS[method], linewidth=2.0,
                    marker=MARKERS[method], markersize=5,
                    linestyle=LINESTYLES[method], label=method, zorder=3)
        ax.axhspan(0.5, 1.02, alpha=0.03, color=bg_color)
        ax.set_xticks(x2); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0.5, 1.02); ax.set_xlim(-0.5, len(idx_list) - 0.5)
        ax.set_xlabel('Preprocessing Configuration')
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    ax_gl.set_ylabel('Mean Jaccard Index')
    ax_gl.legend(loc='lower left', ncol=2, fontsize=8.5)
    fig.suptitle(f'Experiment A — Mean Jaccard Index ({n_meth} methods, 144 frames)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig1b_Jaccard_GL_Med_TwoPanel_{out_date}.png'))

    # ── Fig 2: heatmap method × config ───────────────────────────────────────
    J_matrix = np.array([jac[m].mean(axis=1) for m in methods])   # (n_meth, n_cfg)
    fig, ax = plt.subplots(figsize=(15, max(3.5, n_meth * 0.5)))
    cmap_j = LinearSegmentedColormap.from_list('jac', ['#d73027','#fee090','#4575b4'], N=256)
    vmin, vmax = max(0.5, J_matrix.min() - 0.01), min(1.0, J_matrix.max() + 0.005)
    im = ax.imshow(J_matrix, aspect='auto', cmap=cmap_j, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015).set_label('Mean Jaccard Index', fontsize=10)
    for i in range(n_meth):
        for j in range(n_cfg):
            val = J_matrix[i, j]
            col = 'white' if val < (vmin + vmax) / 2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6.5, color=col, fontweight='bold')
    ax.set_xticks(np.arange(n_cfg)); ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=8.5)
    ax.set_yticks(np.arange(n_meth)); ax.set_yticklabels(methods, fontsize=10)
    for i in range(n_meth):
        bj = int(np.argmax(J_matrix[i]))
        ax.add_patch(plt.Rectangle((bj - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='gold', linewidth=2.5))
    ax.axvline(8.5, color='white', linewidth=1.5, alpha=0.6)
    ax.set_title('Experiment A — Jaccard Index Heatmap (Methods × Preprocessing Configs)\n'
                 'Gold border = best config per method', pad=8)
    ax.grid(False); plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig2_Heatmap_MethodxConfig_{out_date}.png'))

    # ── Fig 3: violin at GL82 ─────────────────────────────────────────────────
    gl82_idx = cfg_names.index('GL82') if 'GL82' in cfg_names else best_idx[0]
    _plot_violin_strip(
        [jac[m][gl82_idx, :] for m in methods], methods,
        title='Experiment A — Jaccard Distribution at GL82 (144 frames)',
        ylabel='Jaccard Index',
        path=os.path.join(OUTPUT_DIR, f'{tag}_Fig3_Violin_GL82_{out_date}.png'),
    )

    # ── Fig 4: violin at best config per method ───────────────────────────────
    best_cfg_per = [int(np.argmax(jac[m].mean(axis=1))) for m in methods]
    data_bc   = [jac[m][best_cfg_per[i], :] for i, m in enumerate(methods)]
    labels_bc = [f"{m}\n({cfg_names[best_cfg_per[i]]})" for i, m in enumerate(methods)]
    _plot_violin_strip(
        data_bc, methods, tick_labels=labels_bc,
        title='Experiment A — Jaccard at Best Config per Method (144 frames)',
        ylabel='Jaccard Index',
        path=os.path.join(OUTPUT_DIR, f'{tag}_Fig4_Violin_BestConfig_{out_date}.png'),
    )

    # ── Fig 5: focal stats lollipop ───────────────────────────────────────────
    _plot_focal_stats(df_focal,
                      path=os.path.join(OUTPUT_DIR, f'{tag}_Fig5_Stats_FocalTest_{out_date}.png'))

    # ── Fig 6: FPS bar chart ──────────────────────────────────────────────────
    fps_vals = []
    for m in methods:
        t = np.nanmean(tim[m][best_idx])
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
    ax.legend(fontsize=9); ax.set_xlim(0, max(fps_vals) * 1.18)
    ax.grid(axis='x', alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig6_FPS_{out_date}.png'))

    # ── Fig 7: pairwise Wilcoxon heatmap ─────────────────────────────────────
    n = n_meth
    pmat = np.ones((n, n))
    for _, row in df_pair.iterrows():
        a, b = row['Method_A'], row['Method_B']
        if a in methods and b in methods:
            i, j = methods.index(a), methods.index(b)
            pv = float(row['p_value']) if not pd.isna(row['p_value']) else 1.0
            pmat[i, j] = pmat[j, i] = pv
    fig, ax = plt.subplots(figsize=(max(7, n * 0.9), max(6, n * 0.85)))
    cmap_p = LinearSegmentedColormap.from_list('pval', ['#2166ac','#92c5de','#f4a582','#d6604d'], N=256)
    im = ax.imshow(np.log10(pmat + 1e-10), vmin=-4, vmax=0, cmap=cmap_p)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('log₁₀(p-value)', fontsize=10)
    cbar.set_ticks([-4,-3,-2,-1,0]); cbar.set_ticklabels(['0.0001','0.001','0.01','0.1','1.0'])
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
                txt = _sig_stars(pv)
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if pv < 0.01 else 'black', fontweight='bold')
    ax.grid(False); plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig7_Pairwise_Wilcoxon_{out_date}.png'))

    # ── Fig 8: summary panel J + AD + FPS ────────────────────────────────────
    # Reorder df_gl to match methods list
    df_gl_ord = df_gl.reindex(methods)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [
        ('Jaccard_mean', 'Mean Jaccard Index',    df_gl_ord['Jaccard_mean'], True),
        ('AD_px',        'Mean AD (pixels)',        df_gl_ord['AD_px'],        False),
        ('FPS',          'Frames per Second',        df_gl_ord['FPS'],          True),
    ]
    for ax, (col, ylabel, vals, higher_better) in zip(axes, metrics):
        colors = [COLORS[m] for m in methods]
        bars = ax.bar(range(n_meth), vals.values, color=colors,
                      edgecolor='white', linewidth=0.5, width=0.7)
        best_i = int(np.argmax(vals) if higher_better else np.argmin(vals))
        bars[best_i].set_edgecolor('gold'); bars[best_i].set_linewidth(2.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f'{val:.3f}' if col != 'FPS' else f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(range(n_meth))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{"↑ higher = better" if higher_better else "↓ lower = better"}',
                     fontsize=9, color='0.5')
        ax.set_xlim(-0.6, n_meth - 0.4)
        ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    fig.suptitle(f'Experiment A — Performance Summary (GL80/GL82/GL84, 144 frames, {n_meth} methods)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'{tag}_Fig8_Summary_Panel_{out_date}.png'))


# ============================================================================
# Experiment B1 figures
# ============================================================================

_B1_METHOD_LABEL = {
    'RHT':       r'RHT - Xu et al.\ \cite{xu1990new}',
    'RCD':       r'RCD - Chen and Chung \cite{chen2001efficient}',
    'RFCA':      r'RFCA - Ladr\'on de Guevara \cite{ladron2011robust}',
    'Nurunnabi': r'Nurunnabi et al.\ \cite{nurunnabi2018robust}',
    'Guo':       r'Guo and Yang \cite{guo2019iterative}',
    'Greco':     r'Greco et al.\ \cite{greco2023impartial}',
    'Qi':        r'Qi et al.\ \cite{qi2024robust}',
    'CIBICA':    r'CIBICA \cite{romancibica}',
    '3C-FBI':    r'3C-FBI',
}


def _export_B1_latex(J_df, outs, out_date):
    """Write Table 2: compare at 4 decimals, bold max per column, display 3."""
    methods = METHODS_B
    n_meth  = len(methods)
    J_mat = np.zeros((n_meth, len(outs)))
    for k, m in enumerate(methods):
        J_mat[k] = J_df.loc[m, [f'{o} outliers' for o in outs]].values.astype(float)
    rows_4       = np.round(J_mat, 4)
    means_4      = np.round(rows_4.mean(axis=1), 4)
    best_per_col = rows_4.max(axis=0)
    best_mean    = means_4.max()
    lines = []
    lines.append(r'\begin{table}[!htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{(Experiment B1) Mean Jaccard index over '
                 r'100 realizations for semicircle fitting (50 points, '
                 r'$\sigma = 1$~mm noise) under varying outlier counts.'
                 r' All methods received identical point sets per realization.'
                 r' Best value(s) per column in bold (4-decimal comparison).}')
    lines.append(r'\label{tab:jaccard_results}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{lccccccc}')
    lines.append(r'\hline')
    lines.append(r'Method \textbackslash $\;$No.\ outliers & ' +
                 ' & '.join(str(o) for o in outs) + r' & Mean \\')
    lines.append(r'\hline')
    for k, m in enumerate(methods):
        label = _B1_METHOD_LABEL.get(m, m)
        cells = []
        for oi, _ in enumerate(outs):
            v = rows_4[k, oi]
            s = f'{v:.3f}'
            cells.append(rf'\textbf{{{s}}}' if v == best_per_col[oi] else s)
        mv = means_4[k]
        ms = f'{mv:.3f}'
        cells.append(rf'\textbf{{{ms}}}' if mv == best_mean else ms)
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')
    path = os.path.join(OUTPUT_DIR, f'B1_Table_Jaccard_{out_date}.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {path}")


def save_B1_figures(d, out_date):
    outs    = d['outs']
    J_df    = d['J']
    AD_df   = d['AD']
    RMSE_df = d['RMSE']
    Tim_df  = d['Tim']
    methods = METHODS_B
    x = np.array(outs)
    _export_B1_latex(J_df, outs, out_date)

    # ── B1 Fig 1: Jaccard vs outlier count (line) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in methods:
        mean = J_df.loc[method, [f'{o} outliers' for o in outs]].values.astype(float)
        ax.plot(x, mean, color=COLORS[method], linewidth=2.0,
                marker=MARKERS[method], markersize=6,
                linestyle=LINESTYLES[method], label=method, zorder=3)
    ax.set_xlabel('Number of Outlier Points')
    ax.set_ylabel('Mean Jaccard Index')
    ax.set_xticks(x); ax.set_ylim(0.3, 1.02)
    ax.set_title(f'Experiment B1 — Robustness to Outliers (Semicircle)\n'
                 f'Center=({B1_X0},{B1_Y0}), r={B1_R0}, n=50, σ=1 mm, 100 trials per condition')
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B1_Fig1_Jaccard_{out_date}.png'))

    # ── B1 Fig 1b: 2×3 scatter, data only (matches V02 Fig 2) ────────────────
    theta_arc = np.linspace(0, np.pi, 300)
    xarc_gt = B1_X0 + B1_R0 * np.cos(theta_arc)
    yarc_gt = B1_Y0 + B1_R0 * np.sin(theta_arc)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()
    for ax, n_out in zip(axes, outs):
        rng_ex = np.random.default_rng(seed=42 + n_out)
        pts    = generate_semicircle_points(B1_X0, B1_Y0, B1_R0,
                                            n_points=50, noise_std=1.0,
                                            n_outliers=n_out, rng=rng_ex)
        dists  = np.abs(np.sqrt((pts[:,0]-B1_X0)**2+(pts[:,1]-B1_Y0)**2)-B1_R0)
        is_out = dists > 5.0
        ax.scatter(pts[~is_out,0], pts[~is_out,1], c='steelblue', s=22, zorder=4)
        if is_out.any():
            ax.scatter(pts[is_out,0], pts[is_out,1],
                       c='crimson', marker='x', s=60, linewidths=1.8, zorder=5)
        ax.plot(xarc_gt, yarc_gt, 'r--', linewidth=1.5, alpha=0.8)
        ax.plot(B1_X0, B1_Y0, 'r+', markersize=12, markeredgewidth=1.5, zorder=6)
        ax.set_title(f'{n_out} outlier{"s" if n_out != 1 else ""}', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-70, 170); ax.set_ylim(-60, 180)
        ax.set_xlabel('X (mm)', fontsize=9); ax.set_ylabel('Y (mm)', fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle('Experiment B1 — Synthetic Semicircle Realizations (0–5 outliers)\n'
                 f'Center=({B1_X0},{B1_Y0}), r={B1_R0} mm, n=50, σ=1 mm',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B1_Fig1b_Scatter_{out_date}.png'))

    # ── B1 Fig 2: three-panel J + AD + RMSE ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panel_data = [
        (J_df,    'Mean Jaccard Index',   (0.3, 1.02)),
        (AD_df,   'Mean Center Error (mm)', None),
        (RMSE_df, 'Mean Radius Error (mm)', None),
    ]
    for ax, (df_p, ylabel, ylim) in zip(axes, panel_data):
        for method in methods:
            mean = df_p.loc[method, [f'{o} outliers' for o in outs]].values.astype(float)
            ax.plot(x, mean, color=COLORS[method], linewidth=2.0,
                    marker=MARKERS[method], markersize=5,
                    linestyle=LINESTYLES[method], label=method)
        ax.set_xlabel('Number of Outlier Points')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        if ylim: ax.set_ylim(*ylim)
    axes[0].legend(ncol=2, fontsize=8)
    axes[2].legend(ncol=2, fontsize=8)
    fig.suptitle('Experiment B1 — Semicircle Performance Summary\n'
                 f'Center=({B1_X0},{B1_Y0}), r={B1_R0}, n=50, σ=1 mm, 100 trials',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B1_Fig2_Panel_{out_date}.png'))


# ============================================================================
# Experiment B2 figures
# ============================================================================

def save_B2_figures(d, out_date):
    J_mean      = d['J_mean']          # (n_meth, nN, nO, nQ)
    wins        = d['wins']
    tim         = d['tim']
    noise_pct   = d['noise_pct']
    outlier_pct = d['outlier_pct']
    q_values    = d['q_values']
    methods     = METHODS_B
    nN, nO, nQ  = len(noise_pct), len(outlier_pct), len(q_values)
    n_meth      = len(methods)
    q_labels    = [str(q) for q in q_values]
    o_labels    = [f'{op}%' for op in outlier_pct]
    ni0         = noise_pct.index(0)
    oi0         = outlier_pct.index(0)
    qi0         = q_values.index(0)
    best        = np.argmax(J_mean, axis=0)      # (nN, nO, nQ)
    total_cells = nN * nO * nQ

    # ── B2 Fig 1: three-panel line plots ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # (a) vs quantization q
    ax = axes[0]
    for k, method in enumerate(methods):
        ax.plot(range(nQ), J_mean[k, ni0, oi0, :],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
    ax.set_xticks(range(nQ)); ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_xlabel('Quantization Step q'); ax.set_ylabel('Mean Jaccard Index')
    ax.set_title('(a) vs Spatial Quantization\n(noise=0%, outliers=0%)'); ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, fontsize=8)
    # (b) vs outlier fraction
    ax = axes[1]
    for k, method in enumerate(methods):
        ax.plot(outlier_pct, J_mean[k, ni0, :, qi0],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
    ax.set_xlabel('Outlier Fraction (%)'); ax.set_title('(b) vs Outlier Fraction\n(noise=0%, q=0 continuous)')
    ax.set_ylim(0, 1.05)
    # (c) vs noise
    ax = axes[2]
    for k, method in enumerate(methods):
        ax.plot(noise_pct, J_mean[k, :, oi0, qi0],
                color=COLORS[method], marker=MARKERS[method], markersize=5,
                linestyle=LINESTYLES[method], linewidth=2.0, label=method)
    ax.set_xlabel('Noise Level (% of radius)'); ax.set_title('(c) vs Noise Level\n(outliers=0%, q=0 continuous)')
    ax.set_ylim(0, 1.05); ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig1_Panel_Lines_{out_date}.png'))

    # ── B2 Fig 2: best-method multi-panel — one per resolution (V02 Fig 4 style) ─
    res_labels_b2 = {0: '∞ (cont.)', 1: '240×240', 2: '120×120',
                     4: '60×60',    8: '30×30',   16: '15×15'}
    _txt_col = {'CIBICA': 'black', '3C-FBI': 'white', 'RHT': 'white', 'RCD': 'black',
                'RFCA':   'black', 'Nurunnabi': 'white', 'Guo': 'black',
                'Greco':  'black', 'Qi':   'black'}
    method_colors = [COLORS[m] for m in methods]
    cmap_meth = ListedColormap(method_colors)
    norm_meth = BoundaryNorm(np.arange(-0.5, n_meth + 0.5), n_meth)
    ncols_b2  = 2; nrows_b2 = (nQ + ncols_b2 - 1) // ncols_b2
    o_tick_labels = [f'{op}%' for op in outlier_pct]
    n_tick_labels = [f'{nv}%' for nv in noise_pct]
    fig, axes = plt.subplots(nrows_b2, ncols_b2,
                             figsize=(5.5 * ncols_b2, 4.5 * nrows_b2))
    axes = axes.flatten()
    for qi, (ax, q) in enumerate(zip(axes, q_values)):
        data_meth = best[:, :, qi]                        # (nN, nO)
        J_best_qi = np.max(J_mean[:, :, :, qi], axis=0)  # (nN, nO)
        ax.imshow(data_meth, aspect='auto', cmap=cmap_meth, norm=norm_meth, origin='lower')
        ax.set_xticks(range(nO)); ax.set_xticklabels(o_tick_labels, fontsize=12, rotation=45)
        ax.set_yticks(range(nN)); ax.set_yticklabels(n_tick_labels, fontsize=12)
        ax.set_xlabel('Outlier contamination', fontsize=13)
        ax.set_ylabel('Noise σ', fontsize=13)
        ax.text(0.5, 1.02, res_labels_b2.get(q, str(q)),
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=14, fontweight='bold')
        for ni2 in range(nN):
            for oi2 in range(nO):
                m_idx = data_meth[ni2, oi2]
                jac_v = J_best_qi[ni2, oi2]
                ax.text(oi2, ni2, f'{methods[m_idx][:5]}\n{jac_v:.3f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color=_txt_col.get(methods[m_idx], 'black'))
        ax.grid(False)
    for ax in axes[nQ:]:
        ax.set_visible(False)
    handles_leg = [mpatches.Patch(color=COLORS[m], label=m) for m in methods]
    n_legend_cols = (n_meth + 1) // 2   # split into two rows
    fig.legend(handles=handles_leg, loc='lower center', ncol=n_legend_cols,
               fontsize=13, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig2_Heatmap_BestMethod_{out_date}.png'))

    # ── B2 Fig 3: Jaccard heatmaps at 3 noise levels ─────────────────────────
    noise_panel_vals = [0, 2, 5]
    ni_panel = [noise_pct.index(n) for n in noise_panel_vals if n in noise_pct]
    fig, axes = plt.subplots(1, len(ni_panel), figsize=(7 * len(ni_panel), 6), sharey=True)
    if len(ni_panel) == 1: axes = [axes]
    cmap_j = LinearSegmentedColormap.from_list('jac2', ['#d73027','#fee090','#4575b4'], N=256)
    for ax, ni in zip(axes, ni_panel):
        best_ni = np.argmax(J_mean[:, ni, :, :], axis=0)
        data    = J_mean[best_ni, ni, np.arange(nO)[:, None], np.arange(nQ)[None, :]]
        im = ax.imshow(data, aspect='auto', cmap=cmap_j, vmin=0, vmax=1)
        ax.set_xticks(range(nQ)); ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_yticks(range(nO)); ax.set_yticklabels(o_labels, fontsize=9)
        ax.set_xlabel('Quantization Step q (coarser →)')
        ax.set_title(f'Noise = {noise_pct[ni]}%', fontsize=11, fontweight='bold')
        for oi2 in range(nO):
            for qi2 in range(nQ):
                ax.text(qi2, oi2, f'{data[oi2,qi2]:.3f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if data[oi2,qi2] < 0.5 else 'black', fontweight='bold')
        ax.grid(False)
    axes[0].set_ylabel('Outlier Fraction p')
    plt.colorbar(im, ax=axes[-1], fraction=0.025, pad=0.02).set_label('Max Jaccard (best method)', fontsize=9)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig3_Heatmap_NoisePanels_{out_date}.png'))

    # ── B2 Fig 4: win count bar chart ─────────────────────────────────────────
    win_counts = [int(wins.loc[m, 'Wins']) if m in wins.index else 0 for m in methods]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, win_counts, color=[COLORS[m] for m in methods],
                  edgecolor='white', linewidth=0.5, width=0.7)
    for bar, w in zip(bars, win_counts):
        pct = 100 * w / total_cells
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_cells * 0.005,
                f'{w}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Configurations Won')
    ax.set_ylim(0, max(win_counts) * 1.2)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig4_WinCount_{out_date}.png'))

    # ── B2 Fig 5: 3C-FBI category multi-panel — one per resolution (V02 Fig 5 style) ─
    cat_bounds = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]
    cat_labels = ['Excellent\n(≥0.99)', 'Very Good\n(≥0.95)', 'Good\n(≥0.90)',
                  'Acceptable\n(≥0.80)', 'Marginal\n(≥0.70)', 'Poor\n(≥0.50)',
                  'Very Poor\n(<0.50)']
    cat_colors = ['#2166ac', '#4dac26', '#b8e186', '#f7f7f7', '#f1a340', '#d7191c', '#7b2d00']
    fbi_idx = methods.index('3C-FBI')

    def _cat_idx(j_val):
        for ci, bound in enumerate(cat_bounds):
            if j_val >= bound: return ci
        return len(cat_bounds)

    n_cats   = len(cat_labels)
    cmap_cat = ListedColormap(cat_colors)
    norm_cat = BoundaryNorm(np.arange(-0.5, n_cats + 0.5), n_cats)
    fig, axes = plt.subplots(nrows_b2, ncols_b2,
                             figsize=(7.5 * ncols_b2, 4.5 * nrows_b2))
    axes = axes.flatten()
    im_cat = None
    for qi, (ax, q) in enumerate(zip(axes, q_values)):
        fbi_j_qi   = J_mean[fbi_idx, :, :, qi]        # (nN, nO)
        cat_matrix = np.vectorize(_cat_idx)(fbi_j_qi)
        im_cat = ax.imshow(cat_matrix, aspect='auto', cmap=cmap_cat,
                           norm=norm_cat, origin='lower')
        ax.set_xticks(range(nO)); ax.set_xticklabels(o_tick_labels, fontsize=12, rotation=45)
        ax.set_yticks(range(nN)); ax.set_yticklabels(n_tick_labels, fontsize=12)
        ax.set_xlabel('Outlier contamination', fontsize=13)
        ax.set_ylabel('Noise σ', fontsize=13)
        ax.text(0.5, 1.02, res_labels_b2.get(q, str(q)),
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=14, fontweight='bold')
        for ni2 in range(nN):
            for oi2 in range(nO):
                j_val = fbi_j_qi[ni2, oi2]; ci = cat_matrix[ni2, oi2]
                ax.text(oi2, ni2, f'{j_val:.3f}', ha='center', va='center', fontsize=12,
                        color='white' if ci in (0, 5, 6) else 'black', fontweight='bold')
        ax.grid(False)
    for ax in axes[nQ:]:
        ax.set_visible(False)
    plt.subplots_adjust(left=0.07, right=0.82, bottom=0.10, top=0.94, hspace=0.55, wspace=0.40)
    cax = fig.add_axes([0.84, 0.15, 0.025, 0.70])
    cbar5 = fig.colorbar(im_cat, cax=cax, ticks=range(n_cats))
    cbar5.ax.set_yticklabels(cat_labels, fontsize=12)
    cbar5.set_label('Performance Category', fontsize=13)
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig5_PerfCategory_3CFBI_{out_date}.png'))

    # ── B2 Fig 6: resolution visualization (V02 Fig 3 style: filled squares) ──
    res_labels = {0: 'infinite', 1: '240 x 240', 2: '120 x 120',
                  4: '60 x 60',  8: '30 x 30',   16: '15 x 15'}
    nQ_vis  = len(q_values)
    ncols_v = 3
    nrows_v = (nQ_vis + ncols_v - 1) // ncols_v
    fig, axes = plt.subplots(nrows_v, ncols_v, figsize=(5 * ncols_v, 4.5 * nrows_v))
    axes = axes.flatten()
    # 70 inliers on the circle + 30 outliers uniformly in the [0, 2r0]^2 region.
    rng_vis   = np.random.default_rng(seed=0)
    n_in_ref, n_out_ref = 70, 30
    theta     = rng_vis.uniform(0, 2 * np.pi, n_in_ref)
    inliers0  = np.column_stack([B2_X0 + B2_R0 * np.cos(theta),
                                 B2_Y0 + B2_R0 * np.sin(theta)])
    outliers0 = rng_vis.uniform([0.0, 0.0],
                                [2 * B2_X0, 2 * B2_Y0],
                                (n_out_ref, 2))
    for ax, q in zip(axes, q_values):
        if q == 0:
            # Continuous: small filled-square markers, no grid concept.
            in_q, out_q = inliers0, outliers0
            cx_p, cy_p, r_p = B2_X0, B2_Y0, B2_R0
            ax_max = 2 * B2_R0
            ax.scatter(in_q[:, 0],  in_q[:, 1],  marker='s', c='blue',
                       s=10, edgecolors='none', zorder=3)
            ax.scatter(out_q[:, 0], out_q[:, 1], marker='s', c='red',
                       s=16, edgecolors='none', zorder=4)
        else:
            in_q  = np.unique(np.round(inliers0  / q).astype(int), axis=0)
            out_q = np.unique(np.round(outliers0 / q).astype(int), axis=0)
            cx_p, cy_p, r_p = B2_X0 / q, B2_Y0 / q, B2_R0 / q
            ax_max = 2 * B2_R0 / q
            # Visual cell size: 1 cell at coarse grids (each rect IS one pixel),
            # scaled up at fine grids (240, 120) so points stay readable.
            # Inliers are ≥6 cells apart on average at R=240, so sz=4 stays
            # non-overlapping in practice.
            sz_by_q = {1: 4, 2: 2, 4: 1, 8: 1, 16: 1}
            sz = sz_by_q[q]
            half = sz / 2.0
            in_rects  = [mpatches.Rectangle((x - half, y - half), sz, sz) for x, y in in_q]
            out_rects = [mpatches.Rectangle((x - half, y - half), sz, sz) for x, y in out_q]
            ax.add_collection(PatchCollection(in_rects,  facecolors='blue',
                                              edgecolors='none', zorder=3))
            ax.add_collection(PatchCollection(out_rects, facecolors='red',
                                              edgecolors='none', zorder=4))
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(cx_p + r_p * np.cos(th), cy_p + r_p * np.sin(th),
                '--', color='dimgray', linewidth=0.9, alpha=0.85, zorder=2)
        ax.set_xlim(0, ax_max); ax.set_ylim(0, ax_max)
        ax.set_aspect('equal'); ax.grid(False)
        ax.text(0.5, 1.02,
                f'Resolution: {res_labels[q]}\nInliers: {len(in_q)}, Outliers: {len(out_q)}',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=10)
    for ax in axes[nQ_vis:]: ax.set_visible(False)
    plt.tight_layout()
    _savefig(os.path.join(OUTPUT_DIR, f'B2_Fig6_ResolutionPanel_{out_date}.png'))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Regenerate 3C-FBI paper figures from CSVs')
    parser.add_argument('--date', default=None,
                        help='CSV date tag (e.g. 20260501). Auto-detected if omitted.')
    parser.add_argument('--out-date', default=None,
                        help='Date tag for output filenames. Defaults to same as --date.')
    args = parser.parse_args()

    csv_date = args.date or find_csv_date('A_Jaccard_3C-FBI')
    out_date = args.out_date or csv_date
    print(f"Reading CSVs dated {csv_date}  →  writing figures dated {out_date}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Experiment A ─────────────────────────────────────────────────────────
    print('\n=== Experiment A ===')
    d_A = load_exp_A(csv_date)
    for methods, tag in [(METHODS_A, 'A'), (METHODS_A_PAPER, 'Ap')]:
        print(f"  --- View '{tag}': {len(methods)} methods ---")
        save_A_figures(d_A, methods, tag, out_date)

    # ── Experiment B1 ────────────────────────────────────────────────────────
    print('\n=== Experiment B1 ===')
    d_B1 = load_exp_B1(csv_date)
    save_B1_figures(d_B1, out_date)

    # ── Experiment B2 ────────────────────────────────────────────────────────
    print('\n=== Experiment B2 ===')
    d_B2 = load_exp_B2(csv_date)
    save_B2_figures(d_B2, out_date)

    print(f'\nDone. All figures in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
