"""
Main Script for Circle Detection Comparison Study (2026 revision)

Compares CIBICA against eight baselines on 144 real clinical frames
x 18 preprocessing configs (9 green-level + 9 median-filter).

Methods:
  CIBICA      — deterministic ballot-inspection sampling + LS refinement
  HOUGH       — OpenCV HoughCircles (classical CHT baseline)
  RHT         — Randomized Hough Transform           (xu1990new)
  RCD         — RANSAC-based circle detection         (chen2001efficient)
  QI          — IRLS hyperaccurate fitting            (qi2024robust)
  RFCA        — Robust Fitting of Circle Arcs         (ladron2011robust)
  GUO         — Taubin + MAD iterative outlier removal (guo2019iterative)
  GRECO       — Trimmed AMLE                          (greco2023impartial)
  NURUNNABI   — Hyper + LTS robust fitting            (nurunnabi2018robust)

NOTE on HOUGH input: Analysis2025.ipynb feeds HoughCircles the GreenCanny
edge image (already Canny-processed), not the raw GreenMask. We match that
convention here so CHT results are consistent with the published notebook.

Usage:
    cd /Users/erc/Documents/3C-FBI-Circle-fitting
    conda activate poseestimation
    python main_CIBICA_2026.py
"""

import math as m
import os
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CCC_FBI import ccc_fbi
from CIBICA import CIBICA
from HOUGH import HOUGH
from preprocessing import (get_preprocessing_configs, preprocess_green_level,
                            preprocess_median_filter)
from RCD import rcd
from RHT import rht
from QI import qi_2024
from RFCA import rfca
from GUO import guo_2019
from GRECO import greco_2022
from NURUNNABI import nurunnabi


# ---------------------------------------------------------------------------
# Jaccard index
# ---------------------------------------------------------------------------

def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """
    Jaccard index (intersection over union) between two circles.

    Returns 0 for non-overlapping circles, 1 for identical circles.
    Reference: https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6
    """
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    if d == 0:
        return min((r1 / r2) ** 2, (r2 / r1) ** 2)

    d1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d)
    d2 = d - d1
    R = max(r1, r2)
    r = min(r1, r2)

    if d >= r1 + r2:          # no overlap
        return 0.0
    elif d <= R - r:           # one circle inside the other
        return (r / R) ** 2
    else:                      # partial overlap
        alpha1 = 2 * m.acos(max(-1.0, min(1.0, d1 / r1)))
        alpha2 = 2 * m.acos(max(-1.0, min(1.0, d2 / r2)))
        intersection = (0.5 * r1 ** 2 * (alpha1 - m.sin(alpha1)) +
                        0.5 * r2 ** 2 * (alpha2 - m.sin(alpha2)))
        union = m.pi * (R ** 2 + r ** 2) - intersection
        return intersection / union


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiments_with_real_data():
    """
    Run all five detection methods on 144 real frames x 18 preprocessing configs.

    Preprocessing configs (from get_preprocessing_configs()):
      - 9 green-level thresholds: GL70, GL72, ..., GL86
      - 9 median-filter sizes:    Med3, Med5, ..., Med19

    Methods:
      CIBICA     — deterministic ballot-inspection sampling + LS refinement
      HOUGH      — OpenCV HoughCircles applied to GreenCanny edge image
      RHT        — Randomized Hough Transform
      RCD        — RANSAC-based circle detection (distance-constrained)
      QI         — IRLS hyperaccurate fitting (Qi 2024)
      RFCA       — Robust Fitting of Circle Arcs (Ladron 2011)
      GUO        — Taubin + MAD iterative (Guo 2019)
      GRECO      — Trimmed AMLE (Greco 2022)
      NURUNNABI  — Hyper + LTS (Nurunnabi 2018)

    Returns
    -------
    dict with keys:
        'Jaccard_<METHOD>'  : ndarray (n_images, n_configs)
        'config_names'      : list[str]
        'filenames'         : list[str]

    Note: HOUGH receives GreenCanny (edge image after auto-Canny) to match
    the convention in Analysis2025.ipynb (Cell 56), not the raw GreenMask.
    """
    ground_truth = pd.read_csv('Ground_Truth.csv')
    filenames = ground_truth['Filename'].tolist()
    configs = get_preprocessing_configs()

    n_images = len(filenames)
    n_configs = len(configs)

    # Result arrays — one per method
    Jaccard_3CFBI     = np.zeros((n_images, n_configs))
    Jaccard_CIBICA    = np.zeros((n_images, n_configs))
    Jaccard_HOUGH     = np.zeros((n_images, n_configs))
    Jaccard_RHT       = np.zeros((n_images, n_configs))
    Jaccard_RCD       = np.zeros((n_images, n_configs))
    Jaccard_QI        = np.zeros((n_images, n_configs))
    Jaccard_RFCA      = np.zeros((n_images, n_configs))
    Jaccard_GUO       = np.zeros((n_images, n_configs))
    Jaccard_GRECO     = np.zeros((n_images, n_configs))
    Jaccard_NURUNNABI = np.zeros((n_images, n_configs))

    print(f"Processing {n_images} images × {n_configs} preprocessing configs")
    print(f"Methods: 3C-FBI, CIBICA, HOUGH, RHT, RCD, QI, RFCA, GUO, GRECO, NURUNNABI")
    print("=" * 70)

    t_start = time.time()

    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']

        # Load images once per frame
        bs_path = os.path.join('black_sphere_ROI', filename + '.png')
        gb_path = os.path.join('green_back_ROI',   filename + '.png')
        BS_crop = cv2.imread(bs_path)
        G_crop  = cv2.imread(gb_path)

        if BS_crop is None:
            print(f"  Warning: could not load {bs_path} — skipping")
            continue

        xmax = BS_crop.shape[1]  # width  (columns)
        ymax = BS_crop.shape[0]  # height (rows)

        for j, cfg in enumerate(configs):
            try:
                # ---- Preprocessing ----------------------------------------
                if cfg['green_level'] is not None:
                    GreenMask, GreenCanny, edgels = preprocess_green_level(
                        BS_crop, cfg['green_level'])
                else:
                    GreenMask, GreenCanny, edgels = preprocess_median_filter(
                        BS_crop, G_crop, cfg['median_size'])

                # ---- 3C-FBI -----------------------------------------------
                # Returns [row, col] — compare as jaccard_circles(YGT, XGT, ...)
                # NOTE: CCC_FBI.py contains a print() statement; suppress with devnull
                if len(edgels) >= 3:
                    try:
                        import io, contextlib
                        with contextlib.redirect_stdout(io.StringIO()):
                            center_3cfbi, r_3cfbi = ccc_fbi(
                                edgels, Nmax=5000, xmax=xmax, ymax=ymax)
                        if r_3cfbi > 0:
                            Jaccard_3CFBI[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_3cfbi[0], center_3cfbi[1], r_3cfbi)
                    except Exception:
                        pass

                # ---- CIBICA -----------------------------------------------
                # Coordinate convention notes:
                # - frames_to_edgepoints returns edgels as (row, col)
                # - CIBICA.py returns (col, row) due to internal swap in LS refinement
                #   => compare as jaccard_circles(XGT, YGT, ...) where XGT=col, YGT=row
                # - All other edgel-based methods return (row, col)
                #   => compare as jaccard_circles(YGT, XGT, ...)
                # - HOUGH (OpenCV) returns (col, row) = (x, y)
                #   => compare as jaccard_circles(XGT, YGT, ...)
                # GT convention: X = col (horizontal), Y = row (vertical)
                if len(edgels) >= 3:
                    x_c, y_c, r_c = CIBICA(edgels, n_triplets=500,
                                            xmax=xmax, ymax=ymax)
                    if not (np.isnan(x_c) or r_c <= 0):
                        Jaccard_CIBICA[i, j] = jaccard_circles(
                            XGT, YGT, RGT, x_c, y_c, r_c)

                # ---- HOUGH ------------------------------------------------
                # Feed GreenCanny (edge image) to match Analysis2025.ipynb
                # Cell 56: HoughCircles(edgel_frames[i][preprocess_ID]*255, ...)
                # OpenCV returns (col, row) = (x, y) — compare with (XGT, YGT) directly.
                x_h, y_h, r_h = HOUGH(GreenCanny, minDist=300, param2=8,
                                       minRadius=5, maxRadius=20)
                if x_h > 0:
                    Jaccard_HOUGH[i, j] = jaccard_circles(
                        XGT, YGT, RGT, x_h, y_h, r_h)

                # ---- RHT --------------------------------------------------
                # threshold=3 tuned for ~50×50 px images (radius ~11 px)
                if len(edgels) >= 3:
                    center_rht, r_rht = rht(edgels, num_iterations=1000,
                                            threshold=3)
                    if r_rht > 0:
                        Jaccard_RHT[i, j] = jaccard_circles(
                            YGT, XGT, RGT,
                            center_rht[0], center_rht[1], r_rht)

                # ---- RCD --------------------------------------------------
                # min_distance=5: tuned down from default 20 (radius is only ~11 px)
                # min_inliers=5:  tuned down from default 10 (small edge sets)
                if len(edgels) >= 4:
                    center_rcd, r_rcd = rcd(edgels, num_iterations=1000,
                                            distance_threshold=2,
                                            min_inliers=5,
                                            min_distance=5)
                    if r_rcd > 0:
                        Jaccard_RCD[i, j] = jaccard_circles(
                            YGT, XGT, RGT,
                            center_rcd[0], center_rcd[1], r_rcd)

                # ---- QI (Qi 2024 IRLS) ------------------------------------
                if len(edgels) >= 3:
                    try:
                        center_qi, r_qi = qi_2024(edgels)
                        if r_qi > 0:
                            Jaccard_QI[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_qi[0], center_qi[1], r_qi)
                    except Exception:
                        pass  # qi_2024 raises ValueError on degenerate input

                # ---- RFCA (Ladron 2011) ------------------------------------
                if len(edgels) >= 3:
                    try:
                        center_rfca, r_rfca = rfca(edgels)
                        if r_rfca > 0:
                            Jaccard_RFCA[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_rfca[0], center_rfca[1], r_rfca)
                    except Exception:
                        pass

                # ---- GUO (Guo 2019) ----------------------------------------
                if len(edgels) >= 3:
                    try:
                        center_guo, r_guo = guo_2019(edgels)
                        if r_guo > 0:
                            Jaccard_GUO[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_guo[0], center_guo[1], r_guo)
                    except Exception:
                        pass

                # ---- GRECO (Greco 2022) ------------------------------------
                if len(edgels) >= 3:
                    try:
                        center_greco, r_greco = greco_2022(edgels)
                        if r_greco > 0:
                            Jaccard_GRECO[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_greco[0], center_greco[1], r_greco)
                    except Exception:
                        pass

                # ---- NURUNNABI (Nurunnabi 2018) ----------------------------
                if len(edgels) >= 3:
                    try:
                        center_nur, r_nur = nurunnabi(edgels)
                        if r_nur > 0:
                            Jaccard_NURUNNABI[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_nur[0], center_nur[1], r_nur)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  Error — image {filename}, config {cfg['name']}: {e}")

        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            print(f"  Processed {i + 1}/{n_images} images  "
                  f"({elapsed:.1f}s elapsed)")

    print("=" * 70)
    print(f"Done in {time.time() - t_start:.1f}s")

    return {
        'Jaccard_3CFBI':     Jaccard_3CFBI,
        'Jaccard_CIBICA':    Jaccard_CIBICA,
        'Jaccard_HOUGH':     Jaccard_HOUGH,
        'Jaccard_RHT':       Jaccard_RHT,
        'Jaccard_RCD':       Jaccard_RCD,
        'Jaccard_QI':        Jaccard_QI,
        'Jaccard_RFCA':      Jaccard_RFCA,
        'Jaccard_GUO':       Jaccard_GUO,
        'Jaccard_GRECO':     Jaccard_GRECO,
        'Jaccard_NURUNNABI': Jaccard_NURUNNABI,
        'config_names':      [c['name'] for c in configs],
        'filenames':         filenames,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_raw_jaccard_csvs(results, output_dir='.'):
    """
    Save one CSV per method: rows = images, columns = preprocessing configs.
    Filenames: Jaccard_CIBICA.csv, Jaccard_HOUGH.csv, etc.
    """
    config_names = results['config_names']
    filenames    = results['filenames']

    methods = ['3CFBI', 'CIBICA', 'HOUGH', 'RHT', 'RCD', 'QI', 'RFCA', 'GUO', 'GRECO', 'NURUNNABI']
    for method in methods:
        key = f'Jaccard_{method}'
        df = pd.DataFrame(
            results[key],
            index=filenames,
            columns=config_names
        )
        df.index.name = 'Filename'
        path = os.path.join(output_dir, f'{key}.csv')
        df.to_csv(path)
        print(f"  Saved: {path}")


def save_summary_csv(results, output_dir='.'):
    """
    Save mean Jaccard per preprocessing config for all methods.
    Companion CSV for the line-plot figure.
    """
    config_names = results['config_names']
    methods = ['3CFBI', 'CIBICA', 'HOUGH', 'RHT', 'RCD', 'QI', 'RFCA', 'GUO', 'GRECO', 'NURUNNABI']

    rows = []
    for j, cfg in enumerate(config_names):
        row = {'config': cfg}
        for method in methods:
            row[f'mean_Jaccard_{method}'] = np.mean(
                results[f'Jaccard_{method}'][:, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'Figure_All_Methods_Comparison.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    '3CFBI':     '#17becf',   # cyan
    'CIBICA':    '#2ca02c',   # green
    'HOUGH':     '#d62728',   # red
    'RHT':       '#1f77b4',   # blue
    'RCD':       '#ff7f0e',   # orange
    'QI':        '#9467bd',   # purple
    'RFCA':      '#8c564b',   # brown
    'GUO':       '#e377c2',   # pink
    'GRECO':     '#7f7f7f',   # grey
    'NURUNNABI': '#bcbd22',   # yellow-green
}

METHODS = ['3CFBI', 'CIBICA', 'HOUGH', 'RHT', 'RCD', 'QI', 'RFCA', 'GUO', 'GRECO', 'NURUNNABI']


def plot_mean_jaccard(results, output_dir='.'):
    """
    Line plot: mean Jaccard index per preprocessing config for all methods.
    Higher = better.
    """
    config_names = results['config_names']
    x = np.arange(len(config_names))

    fig, ax = plt.subplots(figsize=(15, 6))

    for method in METHODS:
        mean_j = np.mean(results[f'Jaccard_{method}'], axis=0)
        ax.plot(x, mean_j, color=COLORS[method], linewidth=2,
                marker='o', markersize=3, label=method)

    ax.set_xlabel('Preprocessing Config', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Jaccard Index', fontsize=12, fontweight='bold')
    ax.set_title('Mean Jaccard Index per Preprocessing Config — All Methods',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1)
    ax.axvline(x=8.5, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6, label='GL / Med boundary')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, 'Figure_All_Methods_Comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_best_config_boxplot(results, output_dir='.'):
    """
    Box plot: Jaccard distribution across all 144 images for each method
    at the best preprocessing config (highest mean Jaccard for CIBICA).
    Also saves companion CSV.
    """
    config_names = results['config_names']

    # Identify best config for CIBICA (used as common reference)
    best_idx = int(np.argmax(np.mean(results['Jaccard_CIBICA'], axis=0)))
    best_cfg = config_names[best_idx]

    data   = [results[f'Jaccard_{m}'][:, best_idx] for m in METHODS]
    labels = METHODS

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, method in zip(bp['boxes'], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jaccard Index', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Jaccard Distribution at Best Config ({best_cfg}) — 144 Frames',
        fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path_fig = os.path.join(output_dir,
                            f'Figure_Boxplot_BestConfig_{best_cfg}.png')
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path_fig}")
    plt.close()

    # Companion CSV
    rows = []
    for method, vals in zip(METHODS, data):
        for v in vals:
            rows.append({'method': method, 'config': best_cfg, 'Jaccard': v})
    df = pd.DataFrame(rows)
    path_csv = os.path.join(output_dir,
                            f'Figure_Boxplot_BestConfig_{best_cfg}.csv')
    df.to_csv(path_csv, index=False)
    print(f"  Saved: {path_csv}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print mean and best-config Jaccard for all methods."""
    config_names = results['config_names']

    print("\n" + "=" * 70)
    print("Summary — Mean Jaccard Index (all images × all configs)")
    print("=" * 70)
    print(f"{'Method':<10}  {'Mean':>8}  {'Std':>8}  {'Best config':>12}  "
          f"{'Best mean':>10}")
    print("-" * 70)

    for method in METHODS:
        J = results[f'Jaccard_{method}']
        overall_mean = np.mean(J)
        overall_std  = np.std(J)
        per_config   = np.mean(J, axis=0)
        best_idx     = int(np.argmax(per_config))
        best_cfg     = config_names[best_idx]
        best_mean    = per_config[best_idx]
        print(f"{method:<10}  {overall_mean:>8.4f}  {overall_std:>8.4f}  "
              f"{best_cfg:>12}  {best_mean:>10.4f}")

    print("=" * 70)

    # Win counts: how often is each method best across image×config cells?
    n_images, n_configs = results['Jaccard_CIBICA'].shape
    all_J = np.stack([results[f'Jaccard_{m}'] for m in METHODS], axis=0)
    # shape: (5, n_images, n_configs)
    best_method = np.argmax(all_J, axis=0)  # (n_images, n_configs)
    total = n_images * n_configs

    print("\nWin counts (best Jaccard per image×config cell):")
    print(f"  Total cells: {total}")
    for k, method in enumerate(METHODS):
        wins = int(np.sum(best_method == k))
        print(f"  {method:<8}: {wins:4d} / {total}  ({100 * wins / total:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Circle Detection — CIBICA vs HOUGH vs RHT vs RCD vs QI (2026)")
    print("=" * 70)
    print()

    # Run all experiments
    results = run_experiments_with_real_data()
    print()

    # Save raw per-method CSVs
    print("Saving raw Jaccard CSVs...")
    save_raw_jaccard_csvs(results)
    print()

    # Save summary CSV (companion for line plot)
    print("Saving summary CSV...")
    save_summary_csv(results)
    print()

    # Generate figures
    print("Generating figures...")
    plot_mean_jaccard(results)
    plot_best_config_boxplot(results)
    print()

    # Console summary
    print_summary(results)


if __name__ == "__main__":
    main()
