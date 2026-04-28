"""
run_cibica_variability.py — CIBICA repeatability study

Runs CIBICA N_REPS times on all 144 frames × 18 configs = 2592 cases.
Reports per-case std and range, then overall mean(std) and mean(range).

Usage:
    conda activate poseestimation
    python run_cibica_variability.py
"""

import math as m
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

from algorithms.CIBICA import CIBICA
from algorithms.preprocessing import (
    get_preprocessing_configs,
    preprocess_green_level,
    preprocess_median_filter,
)

# ── Parameters ──────────────────────────────────────────────────────────────
N_REPS     = 100
N_TRIPLETS = 500
OUTPUT_DIR = 'variability_results'
# ────────────────────────────────────────────────────────────────────────────


def jaccard_circles(x1, y1, r1, x2, y2, r2):
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


def main():
    ground_truth = pd.read_csv('data/Ground_Truth.csv')
    filenames    = ground_truth['Filename'].tolist()
    configs      = get_preprocessing_configs()

    n_images  = len(filenames)
    n_configs = len(configs)
    n_cases   = n_images * n_configs

    print(f"CIBICA variability study: {n_images} images × {n_configs} configs "
          f"= {n_cases} cases × {N_REPS} reps = {n_cases * N_REPS:,} calls")
    print("=" * 70)

    # ── Step 1: Precompute all edgels + ground truth ────────────────────────
    print("Precomputing edgels...")
    cases = []  # list of dicts, one per (image, config) pair
    for i, filename in enumerate(filenames):
        XGT = ground_truth.iloc[i]['X']
        YGT = ground_truth.iloc[i]['Y']
        RGT = ground_truth.iloc[i]['R']

        BS_crop = cv2.imread(os.path.join('data', 'black_sphere_ROI', filename + '.png'))
        G_crop  = cv2.imread(os.path.join('data', 'green_back_ROI',   filename + '.png'))
        if BS_crop is None:
            for j in range(n_configs):
                cases.append(None)
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
                cases.append(None)
                continue

            if len(edgels) < 3:
                cases.append(None)
                continue

            cases.append({
                'edgels': edgels,
                'xmax': xmax, 'ymax': ymax,
                'XGT': XGT, 'YGT': YGT, 'RGT': RGT,
                'filename': filename, 'config': cfg['name'],
            })

    assert len(cases) == n_cases
    valid = sum(1 for c in cases if c is not None)
    print(f"  {valid}/{n_cases} valid cases ({n_cases - valid} skipped)\n")

    # ── Step 2: Run CIBICA N_REPS times per case ───────────────────────────
    # Shape: (n_cases, N_REPS) — flat rows match ground_truth order
    J = np.zeros((n_cases, N_REPS))

    t_start = time.time()
    for rep in range(N_REPS):
        for idx, case in enumerate(cases):
            if case is None:
                continue
            try:
                x_c, y_c, r_c = CIBICA(case['edgels'], n_triplets=N_TRIPLETS,
                                        xmax=case['xmax'], ymax=case['ymax'])
                if not (np.isnan(x_c) or r_c <= 0):
                    J[idx, rep] = jaccard_circles(
                        case['XGT'], case['YGT'], case['RGT'], x_c, y_c, r_c)
            except Exception:
                pass

        elapsed = time.time() - t_start
        rate = (rep + 1) * valid / elapsed
        eta  = (N_REPS - rep - 1) * valid / rate if rate > 0 else 0
        print(f"\r  Rep {rep+1:3d}/{N_REPS}  |  "
              f"{elapsed/60:.1f} min elapsed  |  "
              f"ETA {eta/60:.1f} min  |  "
              f"{rate:.0f} calls/s", end='', flush=True)

    print(f"\n\nTotal time: {(time.time() - t_start)/60:.1f} min\n")

    # ── Step 3: Compute stats ──────────────────────────────────────────────
    std_per_case   = np.std(J, axis=1)    # (2592,)
    range_per_case = np.ptp(J, axis=1)    # (2592,)  ptp = max - min

    # Only compute means over valid cases
    valid_mask = np.array([c is not None for c in cases])
    mean_std   = std_per_case[valid_mask].mean()
    mean_range = range_per_case[valid_mask].mean()

    print("=" * 50)
    print(f"  mean(std)   = {mean_std:.6f}")
    print(f"  mean(range) = {mean_range:.6f}")
    print(f"  max(std)    = {std_per_case[valid_mask].max():.6f}")
    print(f"  max(range)  = {range_per_case[valid_mask].max():.6f}")
    print("=" * 50)

    # ── Step 4: Save ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, 'cibica_jaccard_variability.npy'), J)

    # Save per-case summary CSV
    rows = []
    for idx, case in enumerate(cases):
        rows.append({
            'case_idx':  idx,
            'filename':  case['filename'] if case else '',
            'config':    case['config'] if case else '',
            'mean_J':    np.mean(J[idx]) if case else np.nan,
            'std_J':     std_per_case[idx] if case else np.nan,
            'range_J':   range_per_case[idx] if case else np.nan,
            'min_J':     np.min(J[idx]) if case else np.nan,
            'max_J':     np.max(J[idx]) if case else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'cibica_variability_summary.csv'), index=False)

    print(f"\nSaved: {OUTPUT_DIR}/cibica_jaccard_variability.npy  shape={J.shape}")
    print(f"Saved: {OUTPUT_DIR}/cibica_variability_summary.csv")


if __name__ == '__main__':
    main()
