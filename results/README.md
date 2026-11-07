# Results index — which runs can be cited

**Read this before quoting any number or feature name from this directory.**

Every run below `archive/` and `smote_in_cv/` was computed on a feature table whose
**compositional column names do not describe their contents**. Any statement of the form
"`non-polar_group` was the most important feature" is false as written: that column holds
the *count of small residues*. The fault and its verification are in
[`../docs/03-dataset-audit-findings.md`](../docs/03-dataset-audit-findings.md), Finding 1.

## Status by run

| Directory | Data | Methodology | Citable? |
|---|---|---|---|
| `rebuilt/` | `peptide_features.csv` — **corrected** | 5×5 stratified CV, failure as positive class, length-stratified | **Yes** |
| `smote_in_cv/<ts>/` | `peptide_baza_formatted.csv` — mislabelled | 10×10 stratified CV, SMOTE per-fold inside pipeline, MCC + AUC reported | Metrics usable; **feature names are wrong** |
| `archive/formatted_dataset/<ts>/` | mislabelled | in-CV selection, Friedman / Nemenyi / Wilcoxon-Holm | Superseded by `smote_in_cv/` |
| `archive/balanced_dataset/<ts>/` | `peptide_baza_balanced.csv` | **SMOTE applied before splitting, persisted to disk** | **No — leakage** |
| `archive/text_results/`, `archive/csv_results/` | mixed, earliest era | pre-dates in-CV selection | **No** |

## The three faults, and which runs they touch

**1. Mislabelled compositional columns — every run except `rebuilt/`.**
Eight of nine group columns hold a different quantity than their name says, and the polar,
charged, basic and acidic counts are absent from the data altogether. Metrics computed on
those columns are still *valid as metrics* — the model learned from real numbers — but every
feature *name* in a selected-feature list or importance ranking is unreliable.

**2. Pre-balanced data — `archive/balanced_dataset/` only.**
`peptide_baza_balanced.csv` was produced by applying SMOTE to the whole dataset and writing
it to disk, so synthetic minority points interpolated from training rows appear in test
folds. Its reported minority-class performance is meaningless. Two further problems make
that file unusable even under correct CV: the synthetic sequences were randomly generated
*after* SMOTE produced the feature vectors, so sequence and descriptors describe different
molecules; and interpolating the one-hot `X4_*` and count `X5_*`/`X8_*` columns yields
fractional values corresponding to no peptide.

**3. No length stratification — all runs including `rebuilt/`'s first pass.**
Length-15 peptides are 58% of the dataset and carry 202 of 232 failures. Any ranking that
places `peptide_len` highly is partly reporting which sub-library a peptide came from.
Within that stratum length's AUC is 0.500 by construction, while mean Janin hydrophobicity
holds at 0.740. See [`../docs/02-source-dataset-and-label.md`](../docs/02-source-dataset-and-label.md) §5.

## If you need a number for a report

Use `rebuilt/`. Current best, out-of-fold, failure as the positive class:

| | ALL (13.1% fail) | LEN==15 (19.5% fail) |
|---|---|---|
| Baseline, length + Janin (AUC) | 0.762 | 0.738 |
| Best model (Random Forest, named + X4/X5/X8) | **0.782** | **0.753** |
| Failure-class recall, same model | 0.35 | 0.46 |
| Failure-class recall, Logistic Regression | 0.71 | 0.72 |

The published ceiling for this task is AUC ≈ 0.773, so 0.782 is at the ceiling rather than
past it. Treat anything materially above ~0.85 on this dataset as a leakage signal.

**Do not compare PR-AUC across the two columns** — chance equals the failure rate, not 0.5,
so the stratum's higher absolute PR-AUC mostly reflects its higher base rate.

## Reproducing

```bash
python src/build_dataset.py          # data/processed/peptide_features.csv
python src/evaluate.py               # results/rebuilt/
python src/build_dataset.py --legacy-check   # proves the rebuild matches the old data
```
