"""
make_visual_comparison.py — Visual comparison figure (TODO 8)

Creates a 1×4 figure showing CIBICA vs CHT overlaid on representative frames:
  - Columns 1–2: typical cases (CIBICA clearly wins)
  - Columns 3–4: challenging cases (both methods tested)

Circles:
  - Green dashed  → ground truth
  - Blue solid    → CIBICA (GL82, best config)
  - Red solid     → CHT / HOUGH (GL80, best config)

Jaccard values are printed below each image.

Output:
  ../nordlinglab-grants-publications-2024/Figures/
      Roman2025_Visual_Comparison_<DATE>.pdf   (300 dpi, vector)
      Roman2025_Visual_Comparison_<DATE>.png   (300 dpi)
      Roman2025_Visual_Comparison_<DATE>.csv   (data for reproducibility)

Run from /Users/erc/Documents/3C-FBI-Circle-fitting:
    conda activate poseestimation
    python make_visual_comparison.py
"""

import os
import sys
import random
from itertools import combinations
from datetime import date

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from algorithms.CIBICA import CIBICA
from algorithms.HOUGH import HOUGH
from algorithms.preprocessing import preprocess_green_level

# ── Configuration ────────────────────────────────────────────────────────────

# Frames chosen to represent the range of difficulty.
# Source: CIBICA_results/Jaccard_CIBICA_20260404.csv  (GL82)
#         CIBICA_results/Jaccard_HOUGH_20260404.csv   (GL80)
FRAMES = [
    # (filename_stem,                   subplot_label)
    ('23839779_20200420_Feet_L_S_2', 'Typical — case A\n(CIBICA 0.988 / CHT 0.779)'),
    ('99999990_20200320_Feet_R_S_0',  'Typical — case B\n(CIBICA 0.939 / CHT 0.718)'),
    ('11156926_20200427_Feet_R_S_1',  'Challenging — case C\n(CIBICA 0.747 / CHT 0.638)'),
    ('29478252_20200608_Feet_R_S_2',  'Challenging — case D\n(CIBICA 0.651 / CHT 0.831)'),
]

CIBICA_GL  = 82    # best preprocessing config for CIBICA
CHT_GL     = 80    # best preprocessing config for CHT
N_TRIPLETS = 500   # matches paper (100 fps operating point)

OUTPUT_DIR = '../nordlinglab-grants-publications-2024/Figures'
TODAY      = date.today().strftime('%Y%m%d')

# ── Jaccard helper (duplicated from main_CIBICA_2026.py) ─────────────────────

def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Jaccard index of two circles."""
    if r1 <= 0 or r2 <= 0:
        return 0.0
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if d >= r1 + r2:
        return 0.0
    if d + r2 <= r1:
        return (r2 / r1) ** 2
    if d + r1 <= r2:
        return (r1 / r2) ** 2
    r1s, r2s, ds = r1 ** 2, r2 ** 2, d ** 2
    cos_a = np.clip((ds + r1s - r2s) / (2 * d * r1), -1, 1)
    cos_b = np.clip((ds + r2s - r1s) / (2 * d * r2), -1, 1)
    alpha = 2 * np.arccos(cos_a)
    beta  = 2 * np.arccos(cos_b)
    intersection = r1s * (alpha - np.sin(alpha)) / 2 + r2s * (beta - np.sin(beta)) / 2
    union = np.pi * r1s + np.pi * r2s - intersection
    return float(intersection / union)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    gt_df = pd.read_csv('data/Ground_Truth.csv')
    gt_df.set_index('Filename', inplace=True)

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2),
                             gridspec_kw={'wspace': 0.08})
    csv_rows = []

    for ax, (filename, label) in zip(axes, FRAMES):
        bs_path = f'data/black_sphere_ROI/{filename}.png'

        BS_crop = cv2.imread(bs_path)
        if BS_crop is None:
            print(f'  Warning: {filename} not found — skipping', file=sys.stderr)
            ax.set_visible(False)
            continue

        h, w = BS_crop.shape[:2]

        # ── Preprocess ───────────────────────────────────────────────────────
        # GL82 → CIBICA edgels
        _, _, edgels = preprocess_green_level(BS_crop, CIBICA_GL)
        # GL80 → CHT edge image (uint8, 0–255)
        _, edge_canny_80, _ = preprocess_green_level(BS_crop, CHT_GL)
        edge_u8 = edge_canny_80.astype(np.uint8)

        # ── Run CIBICA (returns col, row) ────────────────────────────────────
        if len(edgels) >= 3:
            xc, yc, rc = CIBICA(edgels, n_triplets=N_TRIPLETS, xmax=w, ymax=h)
        else:
            xc, yc, rc = float('nan'), float('nan'), 0.0

        # ── Run CHT (returns col, row) ───────────────────────────────────────
        xh, yh, rh = HOUGH(edge_u8, minDist=300, param2=8,
                            minRadius=5, maxRadius=20)

        # ── Ground truth (X=col, Y=row) ──────────────────────────────────────
        xgt = float(gt_df.loc[filename, 'X'])
        ygt = float(gt_df.loc[filename, 'Y'])
        rgt = float(gt_df.loc[filename, 'R'])

        # ── Jaccard ──────────────────────────────────────────────────────────
        j_cibica = jaccard_circles(xgt, ygt, rgt, xc,  yc,  rc)
        j_cht    = jaccard_circles(xgt, ygt, rgt, xh,  yh,  rh)

        # ── Plot ─────────────────────────────────────────────────────────────
        image_rgb = cv2.cvtColor(BS_crop, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb, interpolation='nearest',
                  extent=[-0.5, w - 0.5, h - 0.5, -0.5])  # standard image axes

        # Ground truth — green dashed
        gt_patch = plt.Circle((xgt, ygt), rgt,
                               color='#2ca02c', fill=False,
                               linewidth=1.5, linestyle='--')
        ax.add_patch(gt_patch)

        # CIBICA — blue solid
        if rc > 0 and not np.isnan(xc):
            c_patch = plt.Circle((xc, yc), rc,
                                  color='#1f77b4', fill=False,
                                  linewidth=1.5, linestyle='-')
            ax.add_patch(c_patch)

        # CHT — red solid
        if rh > 0:
            h_patch = plt.Circle((xh, yh), rh,
                                  color='#d62728', fill=False,
                                  linewidth=1.5, linestyle='-')
            ax.add_patch(h_patch)

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)   # row 0 at top
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Case label above, Jaccard below
        case_letter = label.split('—')[1].strip().split('\n')[0].strip()
        ax.set_title(case_letter, fontsize=8.5, pad=3)
        ax.set_xlabel(
            f'CIBICA: {j_cibica:.3f}   CHT: {j_cht:.3f}',
            fontsize=7.5, labelpad=3
        )

        # Annotate difficulty
        difficulty = 'Typical' if 'Typical' in label else 'Challenging'
        ax.text(0.03, 0.97, difficulty, transform=ax.transAxes,
                fontsize=7, color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0))

        print(f'  {filename}: CIBICA={j_cibica:.3f}  CHT={j_cht:.3f}')

        csv_rows.append({
            'Filename':       filename,
            'Difficulty':     difficulty,
            'GT_col':         round(xgt, 4),
            'GT_row':         round(ygt, 4),
            'GT_R':           round(rgt, 4),
            'CIBICA_col':     round(float(xc), 4) if not np.isnan(xc) else '',
            'CIBICA_row':     round(float(yc), 4) if not np.isnan(yc) else '',
            'CIBICA_R':       round(float(rc), 4),
            'CIBICA_Jaccard': round(j_cibica, 4),
            'CHT_col':        round(float(xh), 4),
            'CHT_row':        round(float(yh), 4),
            'CHT_R':          round(float(rh), 4),
            'CHT_Jaccard':    round(j_cht, 4),
            'CIBICA_config':  f'GL{CIBICA_GL}',
            'CHT_config':     f'GL{CHT_GL}',
        })

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='#2ca02c', lw=1.5, ls='--', label='Ground truth'),
        Line2D([0], [0], color='#1f77b4', lw=1.5, ls='-',  label=f'CIBICA (GL{CIBICA_GL})'),
        Line2D([0], [0], color='#d62728', lw=1.5, ls='-',  label=f'CHT (GL{CHT_GL})'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.05), borderpad=0.6)

    fig.suptitle('Representative detection results: CIBICA vs CHT',
                 fontsize=9.5, y=1.02)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base     = f'Roman2025_Visual_Comparison_{TODAY}'
    pdf_path = os.path.join(OUTPUT_DIR, base + '.pdf')
    png_path = os.path.join(OUTPUT_DIR, base + '.png')
    csv_path = os.path.join(OUTPUT_DIR, base + '.csv')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    plt.close(fig)

    print(f'\nSaved:')
    print(f'  {pdf_path}')
    print(f'  {png_path}')
    print(f'  {csv_path}')


if __name__ == '__main__':
    random.seed(42)   # reproducible CIBICA sampling
    np.random.seed(42)
    main()
