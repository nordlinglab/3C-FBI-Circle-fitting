"""
run_triplet_sweep.py — CIBICA triplet count impact on speed and accuracy

Runs CIBICA with different n_triplets values (500, 1000, 2000, 5000, 10000)
over 100 iterations on the full 144-frame × 18-config pipeline.

Outputs:
  - CIBICA_results/TripletSweep_<date>.csv   (raw per-iteration results)
  - CIBICA_results/TripletSweep_Table_<date>.csv (summary table for paper)
  - Console: LaTeX-ready table

Usage:
    cd /Users/erc/Documents/3C-FBI-Circle-fitting
    conda activate poseestimation
    python run_triplet_sweep.py
"""

import os
import time
from datetime import date

import cv2
import numpy as np
import pandas as pd

from algorithms.CIBICA import CIBICA
from algorithms.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)

DATE   = date.today().strftime('%Y%m%d')
OUTPUT = 'CIBICA_results'

N_TRIPLETS_LIST = [500, 1000, 2000, 5000, 10000]
N_ITERATIONS    = 100


def jaccard_circles(x1, y1, r1, x2, y2, r2):
    """Jaccard index between two circles."""
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if r1 + r2 <= d:
        return 0.0
    if d + min(r1, r2) <= max(r1, r2):
        small_area = np.pi * min(r1, r2)**2
        big_area   = np.pi * max(r1, r2)**2
        return small_area / big_area
    cos_a = (r1**2 + d**2 - r2**2) / (2 * r1 * d)
    cos_b = (r2**2 + d**2 - r1**2) / (2 * r2 * d)
    cos_a = np.clip(cos_a, -1, 1)
    cos_b = np.clip(cos_b, -1, 1)
    a1 = r1**2 * np.arccos(cos_a)
    a2 = r2**2 * np.arccos(cos_b)
    tri = 0.5 * np.sqrt(max(0, (-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
    intersection = a1 + a2 - tri
    union = np.pi * r1**2 + np.pi * r2**2 - intersection
    return intersection / union if union > 0 else 0.0


def precompute_edgels():
    """Load images and extract edgels once (shared across all iterations)."""
    ground_truth = pd.read_csv('data/Ground_Truth.csv')
    filenames    = ground_truth['Filename'].tolist()
    configs      = get_preprocessing_configs()

    n_images  = len(filenames)
    n_configs = len(configs)

    print(f"Precomputing edgels for {n_images} images × {n_configs} configs...")
    t0 = time.time()

    data = []
    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']

        BS_crop = cv2.imread(os.path.join('data', 'black_sphere_ROI', filename + '.png'))
        G_crop  = cv2.imread(os.path.join('data', 'green_back_ROI',   filename + '.png'))
        if BS_crop is None:
            continue

        xmax = BS_crop.shape[1]
        ymax = BS_crop.shape[0]

        for j, cfg in enumerate(configs):
            try:
                if cfg['green_level'] is not None:
                    _, _, edgels = preprocess_green_level(BS_crop, cfg['green_level'])
                else:
                    _, _, edgels = preprocess_median_filter(BS_crop, G_crop, cfg['median_size'])
            except Exception:
                continue

            if len(edgels) >= 3:
                data.append({
                    'edgels': edgels,
                    'xmax': xmax, 'ymax': ymax,
                    'XGT': XGT, 'YGT': YGT, 'RGT': RGT,
                })

    print(f"  Precomputed {len(data)} valid edgel sets in {time.time()-t0:.1f}s")
    return data


def run_cibica_sweep(data, n_triplets):
    """Run CIBICA on all precomputed edgel sets, return total time and Jaccard array."""
    jaccards = np.zeros(len(data))

    t_start = time.perf_counter()
    for k, d in enumerate(data):
        try:
            x_c, y_c, r_c = CIBICA(d['edgels'], n_triplets=n_triplets,
                                     xmax=d['xmax'], ymax=d['ymax'])
            if not (np.isnan(x_c) or r_c <= 0):
                jaccards[k] = jaccard_circles(d['XGT'], d['YGT'], d['RGT'],
                                              x_c, y_c, r_c)
        except Exception:
            pass
    elapsed = time.perf_counter() - t_start

    return elapsed, jaccards


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    # Precompute edgels once
    data = precompute_edgels()
    n_calls = len(data)

    # Reference Jaccard (N=10000, single run for baseline)
    print(f"\nComputing reference Jaccard with n_triplets=10000...")
    _, ref_jaccards = run_cibica_sweep(data, 10000)
    ref_mean = ref_jaccards.mean()
    print(f"  Reference mean Jaccard: {ref_mean:.6f}")

    # Sweep
    rows = []
    for n_trip in N_TRIPLETS_LIST:
        print(f"\n{'='*60}")
        print(f"n_triplets = {n_trip}  ({N_ITERATIONS} iterations)")
        print(f"{'='*60}")

        times = []
        jaccard_means = []

        for it in range(N_ITERATIONS):
            elapsed, jaccards = run_cibica_sweep(data, n_trip)
            times.append(elapsed)
            jaccard_means.append(jaccards.mean())

            if (it + 1) % 10 == 0:
                print(f"  iter {it+1}/{N_ITERATIONS}  "
                      f"time={elapsed:.2f}s  J={jaccards.mean():.6f}")

        times = np.array(times)
        jaccard_means = np.array(jaccard_means)
        fps_vals = n_calls / times

        for it in range(N_ITERATIONS):
            rows.append({
                'n_triplets': n_trip,
                'iteration':  it + 1,
                'time_s':     times[it],
                'fps':        fps_vals[it],
                'mean_jaccard': jaccard_means[it],
            })

        print(f"  Summary: time={times.mean():.1f}±{times.std():.1f}s  "
              f"fps={fps_vals.mean():.0f}  "
              f"J={jaccard_means.mean():.6f}  "
              f"ΔJ vs 10k = {(jaccard_means.mean() - ref_mean)*1e6:.0f}×10⁻⁶")

    # Save raw results
    df_raw = pd.DataFrame(rows)
    raw_path = os.path.join(OUTPUT, f'TripletSweep_{DATE}.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Summary table
    summary_rows = []
    for n_trip in N_TRIPLETS_LIST:
        mask = df_raw['n_triplets'] == n_trip
        sub = df_raw[mask]
        summary_rows.append({
            'N_triplets':       n_trip,
            'Mean_time_s':      round(sub['time_s'].mean(), 1),
            'Std_time_s':       round(sub['time_s'].std(), 1),
            'Min_time_s':       round(sub['time_s'].min(), 1),
            'Max_time_s':       round(sub['time_s'].max(), 1),
            'FPS':              round(sub['fps'].mean(), 0),
            'Mean_diff_1e6':    round((sub['mean_jaccard'].mean() - ref_mean) * 1e6, 0),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT, f'TripletSweep_Table_{DATE}.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    # Print LaTeX table
    print(f"\n{'='*60}")
    print("LaTeX table:")
    print(f"{'='*60}")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\hline")
    print(r"$N_{\text{triplets}}$ & Total & std & min & max & fps & Abs.\ difference \\")
    print(r" & time [s] &  & [s] & [s] &  & ($\times 10^{-6}$) \\")
    print(r"\hline")
    for r in summary_rows:
        print(f"{r['N_triplets']} & {r['Mean_time_s']} & {r['Std_time_s']} "
              f"& {r['Min_time_s']} & {r['Max_time_s']} "
              f"& {int(r['FPS'])} & {int(r['Mean_diff_1e6'])} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
