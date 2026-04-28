# CLAUDE.md — Project Guide for AI Assistant

## Project Overview

This repository contains two related circle-fitting papers under active journal revision:

1. **CIBICA** — *Circle Identification in Blurry Images using Combinatorial Approach*
   - Submitted to: Machine Vision and Applications
   - Compares CIBICA vs HOUGH, RHT, RCD, QI (5 methods)
   - Main script: `main_CIBICA_2026.py` → output: `CIBICA_results/`

2. **3C-FBI** — *Combinatorial Convolutions for Circle Fitting in Blurry Images*
   - Extends CIBICA paper with 8–9 method comparison + synthetic experiments
   - Main script: `main_3C_FBI.py` → output: `CCC_FBI_results/`

The 3C-FBI results directly address the reviewer feedback on the CIBICA paper (R1.2, R1.5, R2: "no comparison with other methods", "insufficient visualization", "no runtime performance").

---

## Running

```bash
cd /Users/erc/Documents/3C-FBI-Circle-fitting
conda activate poseestimation

python main_CIBICA_2026.py   # ~2–4 hrs
python main_3C_FBI.py        # ~8–14 hrs (B2 is the bottleneck)
```

---

## Repository Structure

```
algorithms/
  CCC_FBI.py        ← 3C-FBI (proposed method, main algorithm)
  CIBICA.py         ← CIBICA (proposed method for CIBICA paper)
  HOUGH.py          ← OpenCV HoughCircles wrapper
  RHT.py            ← Randomized Hough Transform
  RCD.py            ← RANSAC-based circle detection
  RFCA.py           ← Random Forest Circle Analysis
  NURUNNABI.py      ← Fast LTS circle fitting (max_iter=25, h_proportion=0.7)
  GUO.py            ← Taubin + MAD iterative outlier removal (no LMedS)
  GRECO.py          ← Greco 2022
  QI.py             ← Qi 2024 IRLS hyperaccurate fitting
  preprocessing.py  ← 18 preprocessing configs (9 GL + 9 Med)

data/
  Ground_Truth.csv          ← 144 frames: Filename, X (col), Y (row), R
  black_sphere_ROI/         ← 144 cropped sphere images (.png)
  green_back_ROI/           ← 144 background reference images (.png)

main_CIBICA_2026.py   ← CIBICA paper experiments (5 methods, Exp A only)
main_3C_FBI.py        ← 3C-FBI paper experiments (9 methods, Exp A + B1 + B2)
main_CIBICA.py        ← Legacy script (do not use for paper results)
run_ablation.py       ← CIBICA ablation study (3 variants, CIBICA.py unmodified)
ablation.md           ← Ablation rationale, excluded variants, LaTeX text
```

---

## Coordinate Convention — CRITICAL

- **Ground truth**: `X = col` (horizontal), `Y = row` (vertical)
- **Edgels from preprocessing**: shape `(N, 2)` as `[row, col]`
- **CIBICA returns**: `(x_col, y_row, r)` → needs flip to `[y_row, x_col]` when storing as center
- **RHT / RCD / QI return**: `center = [row, col]` → Jaccard as `jaccard_circles(YGT, XGT, ...)`
- **CIBICA / HOUGH return**: `(col, row)` → Jaccard as `jaccard_circles(XGT, YGT, ...)`

---

## Experiment A — Real World

- **Dataset**: 144 frames × 18 preprocessing configs × N methods
- **Preprocessing configs**: GL70, GL72, GL74, GL76, GL78, GL80, GL82, GL84, GL86 (green-level) + Med3, Med5, Med7, Med9, Med11, Med13, Med15, Med17, Med19 (median filter)
- **Reference configs** (Table 1 style): `BEST_GL = ['GL80', 'GL82', 'GL84']`
- **Three result views**: (i) all 18 configs, (ii) best config per method, (iii) GL80/GL82/GL84
- **HOUGH** receives the **binary edge image** (output of preprocessing × 255), NOT raw grayscale — this makes it vary across configs, matching the reference notebook

### Methods in Exp A
- `main_CIBICA_2026.py`: CIBICA, HOUGH, RHT, RCD, QI
- `main_3C_FBI.py`: CIBICA, 3C-FBI, RHT, RCD, RFCA, Nurunnabi, Guo, Greco, Qi

---

## Experiment B1 — Synthetic Semicircle

- Center (50, 60), r = 100 mm, n = 50 points, σ = 1 mm
- Outliers ∈ {0, 1, 2, 3, 4, 5} replacing existing arc points (on-arc model)
- Outlier magnitude: ±9–10σ radially (random sign)
- Search bounds: `xmax = x0 + r0 = 150`, `ymax = y0 = 60`, `rmax = 2*r0 = 200`
- 100 Monte-Carlo iterations per condition

---

## Experiment B2 — Synthetic Full Circle

- Center (120, 120), r = 120 mm, N = 100 points
- Noise σ/r₀ ∈ {0, 1, 2, 5, 10}%
- Outlier % ∈ {0, 10, 20, 30, 40, 50, 60, 70}
- Quantization q ∈ {0, 1, 2, 3, 6, 12, 24, 40}
- Outlier model: on-arc ±5–20σ radially (random sign)
- After quantization: floor division `x0 // q`, search bounds `xmax = x0_q + r0_q`, `ymax = y0_q + r0_q`
- Stores 5 stats per cell: `[min, mean, median, max, std]`
- 100 Monte-Carlo iterations per condition

---

## Algorithm Parameters — Do Not Change

| Algorithm | Key parameters | Source |
|---|---|---|
| CIBICA | `n_triplets=500` | Paper default |
| RHT | `threshold=3` (Exp A), `threshold=5` (Exp B) | Reference notebook cells 23/84 |
| RCD | `distance_threshold=2, min_inliers=5, min_distance=5` | Tuned for small ROI (r=9–14px) |
| Nurunnabi | `max_iter=25, h_proportion=0.7` | Reference notebook cell 27 |
| Guo | `max_iterations=10, threshold=3` | Reference notebook cell 17 (no LMedS) |

---

## Statistical Analysis (Reviewer Requirements)

For each paper, the focal statistical test compares the proposed method vs each baseline:
- **Wilcoxon signed-rank** (two-sided, paired per frame)
- **Hodges-Lehmann estimator** (median of Walsh averages of differences)
- **95% bootstrap CI** (4000 resamples) for HL estimate
- **Rank-biserial correlation** r_rb as effect size
- Scores computed as per-image mean over GL80/GL82/GL84

Output CSVs: `A_Stats_FocalTest_*.csv` (focal), `A_Stats_Pairwise_*.csv` (all pairs)

---

## Output Files

### CIBICA paper (`CIBICA_results/`)
- `Fig1_Jaccard_AllConfigs_*.png/pdf` — line plot, 18 configs
- `Fig2_Heatmap_MethodxConfig_*.png/pdf` — annotated heatmap
- `Fig3_Violin_GL82_*.png/pdf` — violin+strip at GL82
- `Fig3_Violin_BestGL_*.png/pdf` — violin+strip at GL80/82/84
- `Fig4_Stats_FocalTest_*.png/pdf` — lollipop: CIBICA vs baselines (HL + CI + r_rb)
- `Fig5_FPS_*.png/pdf` — FPS bar chart
- `Fig6_Pairwise_Wilcoxon_*.png/pdf` — p-value heatmap
- `Fig7_Summary_Panel_*.png/pdf` — Jaccard + FPS side by side
- `Fig8_JaccardDistance_*.png/pdf` — 1−J per config (matches paper Fig 12 style)
- `Stats_FocalTest_CIBICA_*.csv`, `Stats_Pairwise_*.csv`
- `Table_AllConfigs_*.csv`, `Table_BestGL_*.csv`, `Table_BestConfig_*.csv`

### Ablation (`ablation_results/`)
- `Ablation_Jaccard_<variant>_*.csv` — per-image × per-config Jaccard
- `Ablation_Summary_*.csv` — mean Jaccard per preprocessing config
- `Ablation_Stats_*.csv` — Wilcoxon + HL + 95% CI (full vs ablated)
- `Ablation_Fig1_Line_*`, `Fig2_Violin_*`, `Fig3_Lollipop_*`, `Fig4_FPS_*`

### 3C-FBI paper (`CCC_FBI_results/`)
- `A_Fig1_Jaccard_AllConfigs_*` — Exp A line plot
- `A_Fig2_Heatmap_MethodxConfig_*` — Exp A heatmap (9×18)
- `A_Fig3_Violin_GL82_*`, `A_Fig4_Violin_BestConfig_*` — distributions
- `A_Fig5_Stats_FocalTest_*` — 3C-FBI vs baselines
- `A_Fig6_FPS_*` — speed comparison
- `A_Fig7_Pairwise_Wilcoxon_*` — p-value matrix
- `A_Fig8_Summary_Panel_*` — Jaccard + AD + FPS panel
- `B1_Fig1_Jaccard_*`, `B1_Fig2_Panel_*` — Exp B1
- `B2_Fig1_Panel_Lines_*` — vs Q, vs outliers, vs noise (3-panel)
- `B2_Fig2_Heatmap_BestMethod_*`, `B2_Fig3_Heatmap_NoisePanels_*`
- `B2_Fig4_WinCount_*`
- `A_Timing_Raw_*.csv`, `B1_Timing_*.csv`, `B2_Timing_*.csv` — runtime data

---

## Plot Style Conventions

All figures use shared `plt.rcParams` (set at top of each main script):
- Font: DejaVu Sans, 11pt base
- Each method has a unique: **color** + **marker** + **linestyle** (for B&W compatibility)
- All figures saved as both **PNG** (300 DPI) and **PDF** (vector)
- No top/right spines; grid alpha=0.3

---

## Ablation Study

Script: `run_ablation.py` → output: `ablation_results/`

```bash
python run_ablation.py   # ~same as main_CIBICA_2026.py runtime × 3 variants
```

### Variants (CIBICA.py is NOT modified)

| Variant | What is removed | Implementation |
|---|---|---|
| `full` | Nothing | `CIBICA(..., refinement=True)` |
| `no_refinement` | LS refinement | `CIBICA(..., refinement=False)` built-in flag |
| `no_consensus` | `median_3d` → per-axis `np.median` | Imports `vectorized_XYR` + `LS_circle` directly |

### Why `no_constraints` is excluded

The bounds/radius filtering inside `vectorized_XYR` progressively deletes rows
from `p1` in synchrony with `cx` and `cy`.  Radius is then computed from the
*filtered* `p1` (line 74).  Bypassing from outside produces shape mismatches or
wrong radii — modifying `CIBICA.py` would be required.  See `ablation.md` for
the full justification and LaTeX text.

### Key results (GL80/GL82/GL84, n=144 frames)

| Variant | Mean J | HL vs full | p | verdict |
|---|---|---|---|---|
| full | 0.8984 | — | — | — |
| no_refinement | 0.8799 | +0.021 | p<0.001 *** | refinement is critical |
| no_consensus | 0.8912 | +0.002 | p=0.20 ns | consensus adds stability (↓std), not mean |

### Cross-run CIBICA consistency check (verified 2026-04-04)

| Source | CIBICA mean Jaccard (GL80–84) |
|---|---|
| CIBICA paper | 0.8968 |
| 3C-FBI paper | 0.8969 |
| Ablation (full) | 0.8984 |

Difference ≤ 0.002 — pipeline is stable and reproducible.

---

## Reference Notebook

Original implementation at:
`/Users/erc/Documents/nordlinglab-digitalupdrs/Process_Video_BlackSphereSize/Analysis2025.ipynb`

Key cells:
- Cell 17: `Guo2019_2` — simple Taubin+MAD, no LMedS
- Cell 23: `RHT` with `threshold=5`
- Cell 27: `nurunnabi_fast_circle_fit` with `max_iter=25`
- Cell 79: `generate_semicircle_points2`, `generate_circle_points`
- Cell 82: `lower_resolution` uses `np.round` (not floor)
- Cell 84: B1 search bounds `xmax=x+r=150, ymax=y=60, rmax=2*r=200`
- Cell 97: B2 stores `[min, mean, median, max, std]` per cell
