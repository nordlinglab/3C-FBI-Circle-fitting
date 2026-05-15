# 3C-FBI Paper — ToDo (updated 2026-05-14)

---

## DONE 2026-05-13 / 2026-05-14 — NRS rerun + DOIs + body trims

### B2 resolution grid switched to NRS (6 resolutions)

- `q_values = [0, 1, 2, 4, 8, 16]` → `R_q ∈ {∞, 240, 120, 60, 30, 15}` px
- 360 cells → 240 cells (9 q × 5 noise × 8 outliers → 6 × 5 × 8)
- Resolved the 480-vs-240 inconsistency in the article: code uses `xmax = x0 + r0 = 240`, so the finest discrete grid is 240×240, not 480×480. Article math updated accordingly (`R_q = (240/q)×(240/q)`, centre `(120/q, 120/q)`).
- B2 figures regenerated; figure layouts: Fig 3 (Resolution panel) is 2 rows × 3 cols, Fig 4 and Fig 5 (heatmaps) are 3 rows × 2 cols.
- Fig 3 fix: dashed reference circle now always at true (120, 120, 120) — previously used `int(B2_X0 // q) * q` which floored to (112, 112) at q=16, creating an asymmetric look.
- Fig 3 rendering: switched from scatter+sized-markers to `Rectangle` patches in data coordinates → no overlap regardless of zoom; 4×4 px squares at q=1, 2×2 at q=2, 1×1 at q ≥ 4.

### Fresh numbers (run 2026-05-13)

| Experiment | 3C-FBI | Next | Notes |
| --- | --- | --- | --- |
| A (best-3-GL Jaccard) | 0.895 (4th) | CIBICA 0.897, Qi/Greco 0.896 | FPS = 144 |
| B1 (mean Jaccard) | **0.9913** (1st) | Qi 0.9910, RFCA 0.9909 | 4 of 7 outlier cols won at 4-dec precision |
| B2 (240 cells, win count) | **90 (37.5 %)** | Qi 67 (27.9 %), RFCA 37 (15.4 %) | peak at 30×30 (23/40), 60×60 (21/40), 120×120 (20/40) |

### Bibliography — local file with DOIs

- New `refs_V03.bib` in the article directory (40 entries, 39 with DOIs). V03 now does `\bibliography{refs_V03}` instead of the 8.6 MB master `../LibraryAllReferences.bib`.
- Cleanup of bib quirks: removed `number = {none}` on ref [25] (al2009error), stripped empty fields from ref [38] (ashyani).
- **Outstanding**: ref [38] `ashyani2022digitizationfoot` still "Manuscript in preparation (2022)" — no DOI possible until published. Resolve before final submission.

### Pattern Recognition compliance pass

- Page count: 33 / 35 ✓
- Bibliography in natural form (12 pt, double-spaced) — no compaction tricks needed at current page count
- All B2 figures use `[H]` (strict placement); `\FloatBarrier` added before bibliography (placeins package)
- Two-row legend in Fig 4 (`ncol=5` instead of `ncol=9`) to fit page width
- 40 references (within PR's 35–55 recommendation), no arXiv flooding

### Body trims to make room (8 cuts, ~35 source lines)

- Intro lit-scan: 5 paragraphs → 1 paragraph + citation cluster
- Detection vs fitting prose: 4-sentence pedagogical explanation → 1 sentence
- "Paper Organization" subsection: removed
- Related Work "strengths-and-drawbacks" summary: removed
- Algorithm 1 follow-up paragraph (Xu et al. comparison): removed (still in Related Work + Discussion)
- Parameter Selection: fixed κ=3 to match Algorithm 1 (was inconsistent τ=1)
- Results section preamble: 8 lines → 1 sentence
- Discussion Strengths: 7-sentence number-by-number retell → 1 paragraph
- Ethics block: 12 lines with sub-headings → 1 paragraph (kept IRB number, dates, consent, confidentiality)

### V03 main file structure

- All sections inlined directly (no `\input{Sxx}`); editing S04-Results.tex does NOT propagate. Edit V03 directly.
- 566 → 566 lines (cuts and additions roughly balanced; bibliography compaction tricks removed, body trimmed).

---

## ⚠ CRITICAL — Repository cleanup (before public release / submission)

The repo must be pruned to expose only the published algorithm (`CCC_FBI_v3.py`).

**Remove:**

- `algorithms/CCC_FBI.py` (v1, ablation)
- `algorithms/CCC_FBI_v2.py` (v2, ablation)
- `algorithms/CCC_FBI_ChatGPT.py` and `algorithms/CCC_FBI_Gemini.py` (LLM comparison)
- `main_3C_FBI_V2.py` (ablation runner)
- `compare_implementations.py` and `notebooks/compare_5methods_ABC.ipynb`
- `CCC_FBI_v1_vs_v2_results/`

**Keep:** `algorithms/CCC_FBI_v3.py`, `main_3C_FBI.py`, `regen_figures.py`, `CCC_FBI_results/`, all data, all other algorithms.

> Do NOT execute until explicitly instructed.

---

Updated after the 2026-05-01 paper review pass and the 2026-05-04 figure overhaul + priority-fix pass. The 71-item review was cross-checked against the current `.tex` files; the breakdown below distinguishes (i) items already completed, (ii) agreed fixes still pending, (iii) author's-call style items, and (iv) items I disagree with based on current file state.

---

## DONE 2026-05-04 — figure overhaul + priority review pass

### B2 figures (V02-style multi-panel)

- B2_Fig2 (best method) and B2_Fig5 (3C-FBI category) rewritten in `regen_figures.py` as 3×3 grids — one panel per spatial resolution (∞ → 12×12), x = outlier %, y = noise σ, `origin='lower'`. Fig 5 thresholds: Excellent ≥0.99, Very Good ≥0.95, Good ≥0.90, Acceptable ≥0.80, Marginal ≥0.70, Poor ≥0.50, Very Poor <0.50.
- New PDF/PNGs copied over `Roman2024_Jaccard_Heatmap3_mean.{pdf,png}` and `Roman2024_Jaccard_Heatmap3_mean_V2.{pdf,png}` in the article Figures folder; .tex `\includegraphics` paths unchanged.
- `regen_figures.py` regenerates all 29 figures in ~15 s from the saved CSVs without touching `main_3C_FBI.py`.

### Article tex updates (V03)

- Quantization list now `q ∈ {0,1,2,3,4,6,12,24,40}` (added q=4 / 120×120).
- Win-count narrative refreshed (3C-FBI 137/360 = 38.1 %, RFCA 101/360 = 28.1 %, Qi 50/360 = 13.9 %, CIBICA 19/360 = 5.3 %; remaining five = 14.7 %).
- Table 2 (`tab:WinnerCount`) extended with `120` column; all 81 cells refreshed; Discussion totals 320 → 360.
- Captions for Fig 4 (best-method) and Fig 5 (3C-FBI categories) rewritten to describe the panel layout, axes, and category bands.
- Four verbose B2 paragraphs collapsed (B2 framework setup, resolution definition, inlier-decrease para, 3C-FBI category breakdown) — page count 36 → 35.

### Priority review fixes (62-item list)

- **A1 (DONE)**: title now consistent — abstract (line 90) and §3.2 first sentence (line 244) both say "Combinatorial Convolution-based Circle Fitting **in** Blurry Images" (was "for").
- **B1 ties (item 22, DONE)**: Table 2 (B1 Jaccard) now bolds Qi at "2 outliers" and RFCA + Qi + 3C-FBI at "4" and "5 outliers" — previously only 3C-FBI was bolded despite ties.
- **B1 / Eq label (DONE)**: removed the orphan `\label{eq:geometric_circle_fitting}` (objective in §1) since it was never referenced anywhere.
- **C3 / dash sweep (DONE)**: en-dash U+2013 in "Hough transform–based" → "Hough-transform-based"; em-dash U+2014 (lines 133, 210) → `---`; "Writing – original/review" en-dashes (CRediT) → `--`. ASCII `(0-5 points)` → `(0--5 points)`.
- **C2 / `\times` spacing (DONE)**: standardized every `N\timesN` in the body to `N \times N` (lines 430, 455, 461, 481, 483, 489, etc.) for consistency with the rest of the paper.
- **CRediT accent (items 39/40, DONE)**: line 530 now reads `Esteban Rom\'an Catafau` to match the author block on line 76 (was unaccented).
- **E3 / median blur capitalisation (DONE)**: Fig 1 caption now says "median blur" (and "green-level") instead of "Median Blur" / "Green Level".

PDF compiles clean (35 pages), no `Multiply defined`, no `Undefined`, no remaining U+2013/U+2014 dashes, no unspaced `\times`, no `(N-N points)` ranges.

---

## Already DONE in earlier sessions

### Code (`/Users/erc/Documents/3C-FBI-Circle-fitting/`)

- **K0**: align 3C-FBI triplet filtering with CIBICA (`vectorized_XYR` filters now match) — done 2026-04-28.
- **K1**: add CIBICA to Experiments B1 and B2 — done 2026-04-30 (radius bounds parametrized via `rmin`/`rmax`/`minval`).
- **3C-FBI = v3, cube_size=3** — published variant chosen after the v1-vs-v3-c3-vs-v3-c5 ablation; main_3C_FBI.py wires `algorithms.CCC_FBI_v3.ccc_fbi_v3` with `cube_size=3`.

### Paper — Algorithm 1 (P1–P9)

All draft items completed and inserted. Algorithm 1 in main paper now has:

- Correct step order (vote → top-N peaks → cube scoring → return).
- Triplet filters explicit in main text (collinearity, ±20-px center bounds, radius range).
- 3×3×3 cube notation with `cube_size=3`, `tau=1`.
- Single `\label{alg:main}`, no duplicates.
- References Eq.~\ref{eq:circle_formula} (now defined locally in the main paper, §3.2.1).

### Paper — Result update for 20260501 run

- Abstract, Table 1, Table 2 (B1 + CIBICA row), Table 3 (B2 + CIBICA row), §4.1 statistical paragraph, §4.2.1/§4.2.2 narratives, §5 Discussion all updated.
- Three figures replaced in `Figures/`: `Roman2024_Jaccard_AllConfigs_wCIBICA`, `Roman2024_Jaccard_Heatmap3_mean`, `Roman2024_Jaccard_Heatmap3_mean_V2`.
- B1 supplementary tables (S1, S2 — AD, RMSE) updated with CIBICA row and 20260501 numbers.
- Float placement fixed: `[t]` → `[!htbp]` so algorithm/tables no longer spill out after References.

### Paper — Supplementary fixes

- **S1**: outlier proportions in S2.3 — `{0,10,20,30,40,50,60,70}%`.
- **S2**: mojibake / encoding glitches partly cleaned (some `â` may still remain — see E12 below).

---

## Agreed fixes — pending

### A. Numerical / consistency errors (HIGH priority)

- **A1. Title vs abstract / algorithm caption inconsistency** (main line 73)
  - Title: "3C-FBI: Combinatorial Convolutions for Circle Fitting in Blurry Images"
  - Abstract / algorithm caption: "Combinatorial Convolution-based Circle Fitting in Blurry Images"
  - **Fix**: change the `\title{...}` to match the rest: "Combinatorial Convolution-based Circle Fitting in Blurry Images".

- **A2. Highlight 0.896 ≠ Table 0.895** (main line 112)
  - Highlight item: "competitive accuracy (0.896 Jaccard)".
  - Table 1: 3C-FBI = 0.895, CIBICA = 0.898.
  - **Fix**: change to "competitive accuracy (0.895 Jaccard)" — or add CIBICA's 0.898 in the same bullet.

- **A3. Abstract "matching Qi and Greco" is technically imprecise** (main line 93)
  - Table 1: 3C-FBI = 0.895, Qi = 0.896, Greco = 0.896 → 3C-FBI is 0.001 below.
  - **Fix**: rewrite to "remaining within 0.001 of Qi et al. and Greco et al." or "comparable to Qi et al. and Greco et al. (Δ ≤ 0.001)".

- **A4. Table 1 AD column — incorrect bold** (main line 364)
  - Greco's `\textbf{0.740}` is bolded but Qi's `0.727` is the true minimum; only Qi should be bold.
  - **Fix**: remove `\textbf{...}` around Greco's `0.740`.

- **A5. Abstract / line 95 reads `0.992 ... narrowly leading Qi at 0.991 ... and substantially exceeding the Random Hough Transform at 0.962`**
  - Old text said "0.989 ... at 0.991 ... at 0.963". New B1 RHT mean is 0.962; verify number used in abstract matches Table 2.
  - **Status**: already 0.962 in current text. ✓ Just confirm consistency in next compile.

### B. References / labels

- **B1. Equation 1 (objective function) has no label** (main line 143)
  - Add: `\label{eq:geometric_circle_fitting}` or similar so it can be referenced.

- **B2. Duplicate `\label{eq:circle_formula}` across files** (main line 256, supp line 360)
  - Elsevier compiles main and supp as separate documents, so this works in practice — but rename supp's to be defensive: `\label{eq:supp_circle_formula}`. No body text references the supp's label, so this is safe.

- **B3. Supplementary `\label{fig:spherelabeling}` (line 140) is never referenced**
  - **Fix**: add a `\ref{fig:spherelabeling}` in S1.4 narrative (e.g., "Figure~\ref{fig:spherelabeling} shows representative manual annotation outcomes."), or remove the figure if redundant with Figure S5 (`fig:pointselection`).

### C. LaTeX typography (MEDIUM priority)

- **C1. Replace `$$...$$` with `\[ ... \]`** — Elsevier prefers display-math brackets.
  - Main: line 427.
  - Supp: lines 266, 273, 277.

- **C2. Replace Unicode `×` with `$\times$`** — main only.
  - Lines 439, 486, 487 (e.g., `12×12` → `$12\times12$`, `160×160` → `$160 \times 160$`).
  - Supp is already clean.

- **C3. En-dash in numeric ranges**
  - Main line 344 (caption): `Jaccard 0.84-0.90` → `Jaccard 0.84--0.90`.
  - Main line 389 (caption): `outliers (0-5)` → `outliers (0--5)`.
  - Main line 232: `radius 9–14 px` (Unicode en-dash inside text mode) → `radius $9$--$14$ px`.
  - Supp line 131: same `$9$–$14$\,px` issue.

### D. Supplementary fixes

- **D1. Heading typo + capitalization** (supp line 192)
  - Current: `\subsection{Experiment A: Real-World data from Parkinson's Disease Assesment in Depth}`
  - Fix: `\subsection{Experiment A: Real-World Data from Parkinson's Disease Assessment in Depth}` (`Assesment` → `Assessment`, `data` → `Data`).

- **D2. `semi circles` → `semicircles`** (supp lines 209 and 231 — table captions).
  - Caption rewrite recommended: "Average Distance (AD) for different circle-fitting methods on synthetic semicircles (mm)" / "RMSE on synthetic semicircles (mm)".

- **D3. Empty `\caption{}` on subfigures** (supp lines 148, 153 inside `fig:pointselection`).
  - **Fix**: add descriptive subcaptions, e.g.,
    - 148: `\caption{Perimeter-point selections (good vs poor).}`
    - 153: `\caption{Annotation error $e_4$.}`
  - Then simplify the outer `\caption{...}` to remove the manual `(a)` / `(b)`.

- **D4. Sub-subsection followed immediately by subsection — broken structure**
  - Supp line 303: `\subsubsection{Additional Results for Experiment B2}` is empty then line 305 starts a new `\subsection`. Either:
    - delete the empty `\subsubsection{Additional Results for Experiment B2}`, OR
    - convert lines 305 and 330 to `\subsubsection{...}` so they sit logically under the parent.

- **D5. `Jaccard Index` (5 occurrences) vs main paper's `Jaccard index`**
  - **Fix**: lowercase `index` throughout supp for consistency with main paper.

### E. Narrative / style polish (LOW priority — author's call)

- **E1.** Main line 130 — "Circle detection and fitting is a fundamental task". Defensible as singular discipline; the proposed fix ("are fundamental tasks") is also fine. Author's call.
- **E2.** Main line 205 — "Evolutionary and Hybrid Approaches". The section *does* discuss evolutionary algos (Ayala 2006 genetic, Jia 2011 gradient-based). Title is acceptable.
- **E3.** Main line 338 — "Median Blur" → "median blur" (lowercase: not a proper noun).
- **E4.** Main line 459 — "noise/outliers" → "noise and outlier contamination".
- **E5.** Main line 489 — "Good"/"Acceptable" categories used without definition. Either add a one-line legend ("Good: Jaccard ≥ 0.90; Acceptable: ≥ 0.80") or lowercase.
- **E6.** Main line 495 (caption) — "Color: excellent ... to poor ..." → "Colors indicate performance from excellent ... to poor ..."
- **E7.** Supp line 281 — replaced the broken `\ref{fig:Fig24}` with a hard-coded "Section~4.2.2" reference. Could be tightened to just "Configuration B2 in the main manuscript".
- **E8.** Main lines 420, 424, 431 — "Framework: ...", "Outliers ... uniformly distributed in square", "Resolution $R_q$: ...". Convert to full sentences ("The framework uses a reference circle ...", "Outliers were uniformly distributed in a square ...", "Resolution $R_q$ was defined as ...").
- **E9.** Main line 458 — list resolutions in monotonic order: 240, 160, 80 (current jumps 160, 80, 240).
- **E10.** Main lines 331–332 (itemize) — add full stops at end of each `\item`.
- **E11.** Main Table 1 caption (line 353) — define AD on first use: "Average Distance (AD, pixels)".
- **E12.** `et al.\` vs `et al.` consistency. Mostly correct now (`\` after `al.` enforces correct spacing); sweep both files: `grep -n "et al\\. \\\\cite"` should return zero hits — fix any that surface.
- **E13.** Main line 447 — "were assigned to" → "were resolved by assigning the win to" (minor).
- **E14.** Main lines 149, 154, 159 — when describing the proposed method, use "circle detection and fitting" (or "fitting") consistently with the paper title's "Circle Fitting".

### F. Pre-submission checks (verify with Pattern Recognition style)

- **F1.** `\linenumbers` (main line 121) is active. Pattern Recognition wants line numbers ON for the review draft, OFF for camera-ready. Comment out only at final acceptance.
- **F2.** Section names: "Declaration of Competing Interest" and "Acknowledgments" — Elsevier accepts both; keep current unless the production editor demands a change.
- **F3.** Keywords: "Outlier resistance" — consider "Robust estimation" or "Robust circle fitting" as a more standard term, optional.

---

## DISAGREE / not actionable (with reasoning)

- **Item 8** of the 71-item list — "eq:circle_formula not referenced". **FALSE.** It IS referenced in main paper line 278: `using Eq.~\ref{eq:circle_formula}` (inside Algorithm 1 step 2). The label is needed and used.
- **Item 10** — "alg:main not referenced". **FALSE.** It IS referenced at lines 246 (`Algorithm~\ref{alg:main}`) and 260 (`Algorithm~\ref{alg:main} presents the full pseudocode...`).
- **Item 20** — "Crooping1 → Cropping1 rename". **DON'T.** The image file in `Figures/` is literally named `Roman2024_Crooping1.PNG`. Renaming the `\includegraphics{...}` would break the include unless the file is also renamed. Either keep the typo or rename both — but a rename is risky pre-submission and offers no reader-visible benefit.
- **Items 28–31, 60–62, 64** — meta-claims that all figures/tables are referenced and no `??` / `QI` typos remain. Verified true post-edits; nothing to do.
- **Item 24** — "up-scaled" vs "upscaled". Both are accepted English; current is fine.

---

## Suggested execution order

1. **A1–A4 (numerical fixes)** — must-fix, ~10 minutes total.
2. **C1–C3 (typography sweep)** — `\[ ... \]`, Unicode `×`, en-dashes — ~10 minutes via search-replace.
3. **D1–D5 (supplementary cleanups)** — ~15 minutes.
4. **B1–B3 (label hygiene)** — ~5 minutes.
5. **E* (style polish)** — bundle into one editing pass before submission.
6. **F1 (line numbers off)** — only at final acceptance.

After each batch, recompile both files (`pdflatex` + `bibtex` + `pdflatex` × 2) and confirm zero `Multiply defined`, `Undefined reference`, or `LaTeX Error` lines in the `.log`.
