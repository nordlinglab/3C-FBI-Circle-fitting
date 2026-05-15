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

## ⚠ Repository Cleanup — PENDING (HIGH PRIORITY)

The repository must be cleaned before the paper is submitted or the code is made public.
Only the **published variant** should remain:

- **Keep**: `algorithms/CCC_FBI_v3.py` (published, cube_size=3)
- **Remove**: `algorithms/CCC_FBI.py` (v1, ablation only)
- **Remove**: `algorithms/CCC_FBI_v2.py` (v2, dense vote map, ablation only)
- **Remove**: `algorithms/CCC_FBI_ChatGPT.py` and `algorithms/CCC_FBI_Gemini.py` (comparison only)
- **Remove**: `main_3C_FBI_V2.py` (ablation runner)
- **Remove**: `compare_implementations.py` and `notebooks/compare_5methods_ABC.ipynb` (internal comparison)
- **Remove**: `CCC_FBI_v1_vs_v2_results/` (ablation output folder)
- **Keep all other scripts, data, and results** (`main_3C_FBI.py`, `regen_figures.py`, `CCC_FBI_results/`, etc.)

Do NOT perform this cleanup automatically — wait for explicit instruction.

---

## Repository Structure

```
algorithms/
  CCC_FBI.py        ← 3C-FBI v1 (kept for ablation; not used by main_3C_FBI.py)
  CCC_FBI_v2.py     ← 3C-FBI v2 (dense vote map; ablation only)
  CCC_FBI_v3.py     ← 3C-FBI v3 (PUBLISHED variant, cube_size=3)
  CIBICA.py         ← CIBICA (rmin/rmax/minval now parametrized for B1/B2)
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
main_3C_FBI.py        ← 3C-FBI paper experiments (9 methods, Exp A + B1 + B2; uses v3, cube_size=3)
main_3C_FBI_V2.py     ← v1-vs-v3-c3-vs-v3-c5 ablation (output: CCC_FBI_v1_vs_v2_results/)
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
- Quantization q ∈ {0, 1, 2, 4, 8, 16} (240/q gives res ∞, 240, 120, 60, 30, 15) — **NRS, updated 2026-05-13**
- Outlier model: on-arc ±5–20σ radially (random sign)
- After quantization: floor division `x0 // q`; search bounds `xmax = x0_q + r0_q`, `ymax = y0_q + r0_q`
- **rmin scaling**: `rmin = min(4, max(1, r0_q − 1))` — keeps rmin=4 everywhere except the coarsest grid (q=16, r0_q=7 → rmin=4 still; rule retained as safety net if grid changes).
- Stores 5 stats per cell: `[min, mean, median, max, std]`
- 100 Monte-Carlo iterations per condition (6 q × 5 noise × 8 outliers = **240 cells**)

---

## Algorithm Parameters — Do Not Change

| Algorithm | Key parameters | Source |
|---|---|---|
| 3C-FBI (v3) | `Nmax=5000, top_n=5, cube_size=3` | Validated 2026-05-01 ablation (see below) |
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

## Figure Regeneration (no re-run needed)

`regen_figures.py` rebuilds every publication figure from the saved CSVs in
`CCC_FBI_results/`. ~15 s vs the 8–14 h `main_3C_FBI.py` re-run.

```bash
conda activate poseestimation
python regen_figures.py --date 20260501
```

It loads `A_Jaccard_*`, `A_Stats_*`, `A_Table1_*`, `Ap_*`, `B1_*`, and
`B2_Jaccard_Full_*` / `B2_Table3_WinCount_*` / `B2_Timing_*`, and writes all 29
figures (A×9, Ap×9, B1×3, B2×6, plus extras). It does NOT import `cv2`, so it
runs anywhere matplotlib + numpy + pandas are available.

### B2_Fig2 / B2_Fig5 layout (V02-style, NRS update 2026-05-13)

Both figures are 2×3 grids (6 panels = NRS resolutions ∞, 240, 120, 60, 30, 15) —
**one panel per spatial resolution**, x-axis = outlier %, y-axis = noise σ.
Origin is `lower` so noise=0% sits at the bottom.

- **B2_Fig2** (best method) uses a `ListedColormap` built from `COLORS[m]` so cell colour matches the method's plot colour everywhere. Method-name text colour comes from a hand-tuned `_txt_col` table (luminance-based).
- **B2_Fig5** (3C-FBI categories) uses thresholds `[0.99, 0.95, 0.90, 0.80, 0.70, 0.50]` (Excellent / Very Good / Good / Acceptable / Marginal / Poor / Very Poor). Colorbar is placed via `subplots_adjust(right=0.82) + fig.add_axes([0.84, 0.15, 0.025, 0.70])` — do NOT use `tight_layout` here, it warns and overlaps the rightmost panel.

### Article figures — name mapping

V03 paper figures live in `/Users/erc/Documents/nordlinglab-grants-publications-2024/Figures/`.
The two B2 panels use historical names so the .tex `\includegraphics` calls don't change:

- `Roman2024_Jaccard_Heatmap3_mean.{pdf,png}` ← `B2_Fig2_Heatmap_BestMethod_*`
- `Roman2024_Jaccard_Heatmap3_mean_V2.{pdf,png}` ← `B2_Fig5_PerfCategory_3CFBI_*`

When refreshing for a new run date, just `cp` from `CCC_FBI_results/` over the existing files.

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
| 3C-FBI paper run 2026-05-01 | 0.8979 |
| 3C-FBI paper run 2026-05-03 (rmin fix + q=4 col) | 0.8987 |

Difference ≤ 0.002 — pipeline is stable and reproducible.

---

## 3C-FBI Algorithm Selection (verified 2026-05-01, refined 2026-05-13)

Following the v1 vs v3-cube3 vs v3-cube5 ablation (`main_3C_FBI_V2.py`),
3C-FBI is `algorithms/CCC_FBI_v3.py` with `cube_size=3`. The 2026-05-13 re-run adopts
the **NRS** grid q ∈ {0,1,2,4,8,16} → resolutions {∞, 240, 120, 60, 30, 15} (240 cells total).

### Final results (n=100 MC, dated 20260513 in `CCC_FBI_results/`, NRS rerun)

| Experiment | 3C-FBI rank | Score | Next best |
| --- | --- | --- | --- |
| A — real (best-3-GL Jaccard) | **4th** | 0.895 | CIBICA 0.897, Qi 0.896, Greco 0.896, RFCA 0.889 |
| B1 — semicircle (Mean J, 3-dec / 4-dec compare) | **1st** | 0.9913 | Qi 0.9910, RFCA 0.9909 |
| B2 — **240**-cell win count (NRS grid) | **1st** | **90 (37.5%)** | Qi 67 (27.9%), RFCA 37 (15.4%), RHT 17 (7.1%), CIBICA 10 (4.2%) |

B2 per-resolution (3C-FBI wins): ∞=6, 240=13, 120=20, 60=21, 30=23, 15=7 — peak in the 30–120 range, exactly the "moderate resolution" sweet-spot.

Statistical test (Exp A best-3-GL): 3C-FBI tied with Greco/Qi/RFCA (p > 0.05),
slightly behind CIBICA (HL = -0.003, p = 0.019), dominates RHT/RCD/Nurunnabi/Guo (p < 0.001).

### Why cube_size=3, not 5

- cube_size=5 (CCC_FBI_v3.py default): wins B1 by 0.0004 J, but loses 0.006 J on Exp A real data (real-world noise has non-uniform vote distribution; wider window pulls centroid).
- cube_size=3: wins B2 outright, ties B1, only 0.001 worse than v1 on Exp A.
- main_3C_FBI.py overrides the v3 default by passing `cube_size=3` in `_call_ccc_fbi`. Do not change.

---

## Article / Submission State (2026-05-14)

### File locations

- **Article repo**: `/Users/erc/Documents/nordlinglab-grants-publications-2024/Article_Roman2021_BlackSphere/`
- **Compile target**: `main_Pattern_recognition_submission_V03_260422_wCIBICA.tex`
- **Output PDF**: same name, 33 pages
- **Section files** (S00–S08): exist on disk but the V03 main file **inlines everything** — do not edit `S04-Results.tex` and expect it to flow through. Edit V03 directly.
- **Bibliography**: `refs_V03.bib` (local 40-entry file with 39 DOIs); V03 uses `\bibliography{refs_V03}` — NOT the 8.6 MB master `../LibraryAllReferences.bib`.
- **Figures**: `/Users/erc/Documents/nordlinglab-grants-publications-2024/Figures/` — three article-named files (`Roman2024_Jaccard_Heatmap3_mean.{pdf,png}`, `Roman2024_Jaccard_Heatmap3_mean_V2.{pdf,png}`, `Roman2024_Resolution2.{pdf,png}`) are copies of `B2_Fig{2,5,6}` from `CCC_FBI_results/`. Refresh with `cp` when `regen_figures.py` rebuilds.

### Compile sequence

```bash
cd /Users/erc/Documents/nordlinglab-grants-publications-2024/Article_Roman2021_BlackSphere
pdflatex -interaction=nonstopmode main_Pattern_recognition_submission_V03_260422_wCIBICA.tex
bibtex   main_Pattern_recognition_submission_V03_260422_wCIBICA
pdflatex -interaction=nonstopmode main_Pattern_recognition_submission_V03_260422_wCIBICA.tex
pdflatex -interaction=nonstopmode main_Pattern_recognition_submission_V03_260422_wCIBICA.tex
```

### Pattern Recognition style — checklist

- Numeric citations (`elsarticle-num`) ✓
- 40 references, all but one with DOIs ✓
- 33 pages ≤ 35 ✓
- All figures use `[H]` (strict placement) inside Results; `\FloatBarrier` before bibliography
- Bibliography is in its natural form (12 pt, double-spaced) — no compaction tricks needed at current page count

### Outstanding journal-compliance items

- **Ref [38] `ashyani2022digitizationfoot`** — "Manuscript in preparation (2022)". Either replace with a current citation or remove before final submission. No DOI possible until published.
- Optional polish: standardize journal-name capitalization across `refs_V03.bib` entries (mixed lowercase/Title Case currently).

### B1 Table 2 generation

The B1 Jaccard table is auto-generated by `regen_figures.py` (via `_export_B1_latex` in both `regen_figures.py` and `main_3C_FBI.py`) as `B1_Table_Jaccard_<DATE>.tex`. **Bolding rule**: 3-decimal display, 4-decimal comparison — only the unique 4-decimal winner per column gets bold; ties at 4 decimals all bold. Currently embedded inline in V03; if you regen B1 numbers, copy the new .tex content over.

### Body-trim record (2026-05-14)

The body was tightened by ~35 source lines to absorb bibliography back-pressure without resorting to font-size shrinks:

- Intro lit-scan compressed to one paragraph + citations
- Detection-vs-fitting prose reduced to one sentence
- "Paper Organization" subsection removed
- Related Work generic "strengths-and-drawbacks" summary removed
- Algorithm 1 follow-up paragraph (Xu et al. comparison) removed
- Parameter Selection updated κ=3 (was τ=1, inconsistent with pseudocode)
- Results section preamble compressed
- Discussion "Strengths" paragraph compressed to single paragraph
- Ethics block compressed (kept IRB number + dates + consent + confidentiality in 1 paragraph)

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
