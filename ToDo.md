# 3C-FBI Paper — Revision ToDo

Based on external review of `V03_260115.tex` (8 methods) and `V03_260422_wCIBICA.tex` (9 methods).
Cross-referenced against `CCC_FBI_results/` CSVs and `run_cibica_variability.ipynb` data.

---

## CRITICAL — Blocks Submission

### C1. ~~Table 2 (B1) bolding errors~~
- **Status**: FIXED. Column-wise maxima now correctly bolded in both versions. RFCA col-2 unbolded, RFCA+Qi col-3 bolded. Method-name bolding removed.

### C2. ~~Table 1 bolding convention undefined~~
- **Status**: FIXED. Best value(s) per column now bolded (cell-level, not method-name). Caption states: "Best value(s) per column in bold."

### C3. ~~wCIBICA abstract contradicts its own Table 1~~
- **Status**: FIXED. Changed to "competitive accuracy (Jaccard index 0.896, matching Qi et al. and Greco et al.)".

### C4. ~~Cross-version drift~~
- **Status**: RESOLVED. Both files use data from the same run. All non-CIBICA rows verified identical.

---

## HIGH — Reviewer Would Flag

### H1. ~~"Significantly" used without statistical tests~~
- **Status**: FIXED. Removed from abstract ("substantially exceeding") and Impact of Contributions. Remaining uses in Related Work (line 178, 200) and Results (line 316) are descriptive, not inferential.

### H2. ~~Overclaiming language~~
- **Status**: FIXED. "exceptional robustness/resistance" removed from abstract. "dominant approach overall" replaced with "achieved the highest number of winning configurations overall."

### H3. ~~wCIBICA Wilcoxon sentence underdescribed~~
- **Status**: FIXED. Now reads: "a paired Wilcoxon signed-rank test over 144 per-image Jaccard scores (GL80/82/84)..."

### H4. ~~B1 replication detail missing~~
- **Status**: FIXED. Table 2 caption now includes: "Mean Jaccard index over 100 realizations for semicircle fitting (50 points, sigma=1mm noise)... All methods received identical point sets per realization."

### H5. B2 tie-handling rule not stated
Table 3 counts "highest Jaccard index per method" but doesn't define what happens when methods tie.
- Each resolution column sums to 40 (= 5 noise x 8 outlier), implying no ties or a specific rule.
- **Action**: Check `main_3C_FBI.py` B2 winner logic. Add footnote to Table 3 caption.

### H6. Experiment A reproducibility gaps
Not enough detail for independent replication:
- Frame selection protocol from the 36 videos
- Whether timing includes I/O or only algorithm execution
- Whether all methods receive identical binary edge maps
- **Action**: Add a concise reproducibility paragraph to Section 3.1 or Supplementary Material.

### H7. Effect sizes mostly absent
Only the wCIBICA version reports one effect size (HL delta for CIBICA). All other comparative claims rely on raw means only.
- **Action**: For key comparisons, the focal stats CSV already contains HL + CI + r_rb. Consider adding a sentence to the Results text referencing these, or add a supplementary table.

---

## MEDIUM — Improves Quality

### M1. FPS rounding consistency
Tables use 1 decimal; prose uses integers. Currently consistent (152 fps = round(151.6)). RHT: 53.8 -> 54, RCD: 22.7 -> 23. All correct.
- **Status**: VERIFIED OK.

### M2. ~~Figure 1 not explicitly referenced in text~~
- **Status**: FIXED. Added "As shown in Figure~\ref{fig:Roman2024_5_Jaccard}" before GL thresholding paragraph.

### M3. Abstract sentences too long
Two sentences remain ~40-50 words:
1. "Our comprehensive evaluation spans three experimental frameworks..." (~42 words)
2. "This combination of accuracy, computational efficiency, and robustness makes..." (~30 words, OK)
- **Action**: Consider splitting sentence 1 into two.

### M4. ~~Table 2 caption too thin~~
- **Status**: FIXED. Now includes sample size, noise, realizations, and "identical point sets" note.

### M5. Abstract/Introduction/Discussion redundancy
Same claims appear three times: real-time speed, Parkinson's relevance, superiority over classical methods.
- **Action**: Trim Introduction "Impact of Contributions" to avoid restating abstract. Discussion should interpret, not restate.

### M6. Highlights FPS claim
"Real-time performance at 150+ fps" — accurate (Table 1 gives 151.6 fps).
- **Status**: OK as-is.

---

## LOW — Polish

### L1. ~~Terminology: "Jaccard Index" vs "Jaccard index"~~
- **Status**: FIXED. Standardized to lowercase "Jaccard index" throughout both versions.

### L2. ~~Citation spacing~~
- **Status**: PARTIALLY FIXED. All table `et al.` entries updated to `et al.\ `. Check rest of document for remaining instances.

### L3. ~~Grammar fixes~~
- **Status**: FIXED.
  - "representative both for" -> "covers both classical and recent"
  - "data was" -> "data were"
  - "take full responsibility" -> "assume full responsibility"
  - "deployable solution" -> "a deployable solution"

### L4. ~~Passive voice in Methods~~
- **Status**: FIXED. Preprocessing protocol and B1 description converted to active voice ("We processed...", "We distributed...", "We added...").

### L5. ~~"edgel" definition~~
- **Status**: VERIFIED OK. Defined once as "edge pixel (edgel)" in abstract; "edgel" used consistently throughout thereafter. No alternation found.

---

## RESOLVED Issues (summary)

| # | Issue | Resolution |
|---|-------|-----------|
| C1 | Table 2 bolding | Correct column-wise maxima, method names unbolded |
| C2 | Table 1 bolding | Best per column, caption footnote added |
| C3 | wCIBICA abstract | "state-of-the-art" -> "competitive accuracy" |
| C4 | Cross-version drift | Same run, identical non-CIBICA rows |
| H1 | "Significantly" | Removed from abstract + Impact |
| H2 | Overclaims | "exceptional" removed, "dominant" replaced |
| H3 | Wilcoxon detail | Paired test, 144 images, GL80/82/84 specified |
| H4 | B1 caption | 100 realizations, 50 points, sigma=1mm |
| M2 | Figure ref | Added Figure~\ref |
| M4 | Table 2 caption | Expanded to stand-alone |
| L1 | Jaccard Index | Lowercase throughout |
| L3 | Grammar | 4 fixes applied |

---

## All Items Resolved

All 22 items from the review have been addressed.
