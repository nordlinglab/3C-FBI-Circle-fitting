# CIBICA Ablation Study

## Why ablation is needed

Reviewer feedback (R1.2, R2) requested evidence that each component of CIBICA
contributes meaningfully to performance.  CIBICA has three distinct processing
stages:

1. **Geometric constraints** — bounds and radius-size filtering inside
   `vectorized_XYR` that discard hypotheses outside the expected image region.
2. **Mode-based consensus** — `median_3d` encodes each (cx, cy, r) triplet into
   a scalar identifier and selects the mode, providing a single robust estimate
   from the cloud of triplet-fitted circles.
3. **Least-squares refinement** — `LS_circle` re-fits a circle to the subset of
   edgels nearest the consensus estimate, improving localization accuracy.

Without ablation, a reviewer cannot distinguish whether performance comes from
the algorithm as a whole or from one dominant stage.  The ablation isolates each
stage by running a variant in which that stage is removed or replaced, while
keeping everything else identical.

---

## When ablation cannot be performed (and why)

### The `no_constraints` variant is not independently implementable

The geometric constraint filtering in CIBICA is structurally coupled to the
radius computation inside `vectorized_XYR`.  After computing the circle centres
(`cx`, `cy`) for all triplets, the function progressively deletes rows from
`p1`—the first point of each triplet—in exact synchrony with deletions from
`cx` and `cy` across four sequential filtering passes (lines 52–71 of
`algorithms/CIBICA.py`).  The radius is then computed on line 74 as:

```python
radius = np.sqrt((cx - p1[:, 0])**2 + (cy - p1[:, 1])**2)
```

Because `p1` is the *filtered* array at this point, its length matches `cx` and
`cy` exactly.  Bypassing the constraint filters from outside the function (for
example, by passing artificially large `xmax`/`ymax` bounds) would suppress
spatial filtering but would not reproduce a true constraint-free execution
path—the structural coupling between `p1` deletions and centre deletions means
any selective modification still requires altering the internals of
`vectorized_XYR`.  Duplicating the function with constraints removed would
effectively create a parallel implementation, introducing maintenance risk and
deviating from the principle of ablating the published algorithm.

For these reasons, the `no_constraints` variant was excluded from the ablation
study.  The two independent components—mode-based consensus and least-squares
refinement—are the ablatable stages and are evaluated instead.

---

## Ablated variants

| Variant | What is removed | Implementation |
|---|---|---|
| `full` | Nothing (complete CIBICA) | `CIBICA(..., refinement=True)` |
| `no_refinement` | Least-squares refinement | `CIBICA(..., refinement=False)` — built-in flag |
| `no_consensus` | Mode-based consensus (`median_3d`) | Imports `vectorized_XYR` + `LS_circle` directly; replaces `median_3d` with `np.median` per axis |

`algorithms/CIBICA.py` is **not modified** by any variant.

---

## Running the ablation

```bash
cd /Users/erc/Documents/3C-FBI-Circle-fitting
conda activate poseestimation
python run_ablation.py
```

Outputs are written to `ablation_results/`:

| File | Content |
|---|---|
| `Ablation_Jaccard_<variant>_<DATE>.csv` | Per-image × per-config Jaccard index |
| `Ablation_Summary_<DATE>.csv` | Mean Jaccard per preprocessing config |
| `Ablation_Stats_<DATE>.csv` | Wilcoxon + Hodges-Lehmann + 95% CI (full vs ablated) |
| `Ablation_Fig1_Line_<DATE>.png/pdf` | Mean Jaccard per preprocessing config |
| `Ablation_Fig2_Violin_<DATE>.png/pdf` | Distribution comparison (GL80/GL82/GL84) |
| `Ablation_Fig3_Lollipop_<DATE>.png/pdf` | HL ± CI: contribution of each component |
| `Ablation_Fig4_FPS_<DATE>.png/pdf` | Processing speed per variant |

Statistical analysis uses Wilcoxon signed-rank (two-sided, paired per frame),
Hodges-Lehmann estimator, and 4000-resample bootstrap 95% CI, consistent with
the main paper's focal test methodology.
