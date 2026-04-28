"""
Main Script for Circle Detection Comparison Study

Compares CIBICA and Hough Transform against RHT, RCD, and QI on real
clinical frames × 18 preprocessing configs (9 green-level + 9 median-filter).

Methods:
  CIBICA  — deterministic ballot-inspection sampling + LS refinement
  HOUGH   — OpenCV HoughCircles (classical CHT baseline)
  RHT     — Randomized Hough Transform           (xu1990new)
  RCD     — RANSAC-based circle detection         (chen2001efficient)
  QI      — IRLS hyperaccurate fitting            (qi2024robust)

Usage:
    python main_CIBICA.py
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

from algorithms import (
    CIBICA, HOUGH, rht, rcd, qi_2024,
    get_preprocessing_configs, preprocess_green_level, preprocess_median_filter
)


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

METHODS = ['CIBICA', 'HOUGH', 'RHT', 'RCD', 'QI']

COLORS = {
    'CIBICA': '#2ca02c',   # green
    'HOUGH':  '#d62728',   # red
    'RHT':    '#1f77b4',   # blue
    'RCD':    '#ff7f0e',   # orange
    'QI':     '#9467bd',   # purple
}


def run_experiments_with_real_data():
    """
    Run all five detection methods on real frames × 18 preprocessing configs.

    Preprocessing configs (from get_preprocessing_configs()):
      - 9 green-level thresholds: GL70, GL72, ..., GL86
      - 9 median-filter sizes:    Med3, Med5, ..., Med19

    Coordinate convention:
      - GT: X = col (horizontal), Y = row (vertical)
      - CIBICA returns (col, row) → compare as jaccard_circles(XGT, YGT, ...)
      - HOUGH (OpenCV) returns (col, row) → compare as jaccard_circles(XGT, YGT, ...)
      - RHT, RCD, QI return (row, col) → compare as jaccard_circles(YGT, XGT, ...)
    """
    ground_truth = pd.read_csv('data/Ground_Truth.csv')
    filenames = ground_truth['Filename'].tolist()
    configs = get_preprocessing_configs()

    n_images = len(filenames)
    n_configs = len(configs)

    Jaccard_CIBICA = np.zeros((n_images, n_configs))
    Jaccard_HOUGH  = np.zeros((n_images, n_configs))
    Jaccard_RHT    = np.zeros((n_images, n_configs))
    Jaccard_RCD    = np.zeros((n_images, n_configs))
    Jaccard_QI     = np.zeros((n_images, n_configs))

    print(f"Processing {n_images} images × {n_configs} preprocessing configs")
    print(f"Methods: {', '.join(METHODS)}")
    print("=" * 70)

    t_start = time.time()

    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']

        bs_path = os.path.join('data', 'black_sphere_ROI', filename + '.png')
        gb_path = os.path.join('data', 'green_back_ROI',   filename + '.png')
        BS_crop = cv2.imread(bs_path)
        G_crop  = cv2.imread(gb_path)

        if BS_crop is None:
            print(f"  Warning: could not load {bs_path} — skipping")
            continue

        xmax = BS_crop.shape[1]
        ymax = BS_crop.shape[0]

        for j, cfg in enumerate(configs):
            try:
                # ---- Preprocessing ----------------------------------------
                if cfg['green_level'] is not None:
                    _, GreenCanny, edgels = preprocess_green_level(
                        BS_crop, cfg['green_level'])
                else:
                    _, GreenCanny, edgels = preprocess_median_filter(
                        BS_crop, G_crop, cfg['median_size'])

                # ---- CIBICA -----------------------------------------------
                if len(edgels) >= 3:
                    x_c, y_c, r_c = CIBICA(edgels, n_triplets=500,
                                            xmax=xmax, ymax=ymax)
                    if not (np.isnan(x_c) or r_c <= 0):
                        Jaccard_CIBICA[i, j] = jaccard_circles(
                            XGT, YGT, RGT, x_c, y_c, r_c)

                # ---- HOUGH ------------------------------------------------
                x_h, y_h, r_h = HOUGH(GreenCanny, minDist=300, param2=8,
                                       minRadius=5, maxRadius=20)
                if x_h > 0:
                    Jaccard_HOUGH[i, j] = jaccard_circles(
                        XGT, YGT, RGT, x_h, y_h, r_h)

                # ---- RHT --------------------------------------------------
                if len(edgels) >= 3:
                    center_rht, r_rht = rht(edgels, num_iterations=1000,
                                            threshold=3)
                    if r_rht > 0:
                        Jaccard_RHT[i, j] = jaccard_circles(
                            YGT, XGT, RGT,
                            center_rht[0], center_rht[1], r_rht)

                # ---- RCD --------------------------------------------------
                if len(edgels) >= 4:
                    center_rcd, r_rcd = rcd(edgels, num_iterations=1000,
                                            distance_threshold=2,
                                            min_inliers=5,
                                            min_distance=5)
                    if r_rcd > 0:
                        Jaccard_RCD[i, j] = jaccard_circles(
                            YGT, XGT, RGT,
                            center_rcd[0], center_rcd[1], r_rcd)

                # ---- QI ---------------------------------------------------
                if len(edgels) >= 3:
                    try:
                        center_qi, r_qi = qi_2024(edgels)
                        if r_qi > 0:
                            Jaccard_QI[i, j] = jaccard_circles(
                                YGT, XGT, RGT,
                                center_qi[0], center_qi[1], r_qi)
                    except Exception:
                        pass  # qi_2024 raises ValueError on degenerate input

            except Exception as e:
                print(f"  Error — image {filename}, config {cfg['name']}: {e}")

        if (i + 1) % 20 == 0 or (i + 1) == n_images:
            elapsed = time.time() - t_start
            print(f"  Processed {i + 1}/{n_images} images  "
                  f"({elapsed:.1f}s elapsed)")

    print("=" * 70)
    print(f"Done in {time.time() - t_start:.1f}s")

    return {
        'Jaccard_CIBICA': Jaccard_CIBICA,
        'Jaccard_HOUGH':  Jaccard_HOUGH,
        'Jaccard_RHT':    Jaccard_RHT,
        'Jaccard_RCD':    Jaccard_RCD,
        'Jaccard_QI':     Jaccard_QI,
        'config_names':   [c['name'] for c in configs],
        'filenames':      filenames,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, output_dir='results/figures'):
    """Line plot of mean Jaccard index per preprocessing config for all methods."""
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
    path = os.path.join(output_dir, 'Figure_Comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    config_names = results['config_names']

    print("\n" + "=" * 70)
    print("Summary — Mean Jaccard Index (all images × all configs)")
    print("=" * 70)
    print(f"{'Method':<8}  {'Mean':>8}  {'Std':>8}  {'Best config':>14}  {'Best mean':>10}")
    print("-" * 70)

    for method in METHODS:
        J = results[f'Jaccard_{method}']
        per_config = np.mean(J, axis=0)
        best_idx = int(np.argmax(per_config))
        print(f"{method:<8}  {np.mean(J):>8.4f}  {np.std(J):>8.4f}  "
              f"{config_names[best_idx]:>14}  {per_config[best_idx]:>10.4f}")

    print("=" * 70)

    # Win counts
    all_J = np.stack([results[f'Jaccard_{m}'] for m in METHODS], axis=0)
    best_method = np.argmax(all_J, axis=0)
    total = best_method.size

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
    print("Circle Detection Comparison: CIBICA vs HOUGH vs RHT vs RCD vs QI")
    print("=" * 70)
    print()

    results = run_experiments_with_real_data()
    print()

    print("Generating figures...")
    plot_results(results)
    print()

    print_summary(results)


if __name__ == "__main__":
    main()
