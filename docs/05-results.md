# Results — predicting Fmoc peptide synthesis success

Reproduction and extension of Gutman et al. 2022, *Predicting the Success of Fmoc-Based
Peptide Synthesis*, ACS Omega, using the dataset published as that paper's supporting
information.

Every number below is reproducible from this repository; the commands are in §8.

---

## 1. Summary of findings

1. **The published per-feature benchmark reproduces.** Computing the paper's own residue
   groupings independently gives AUC 0.483 for aromatic content and 0.502 for polar content,
   against 0.487 and 0.497 reported. The pipeline measures what the paper measured.

2. **The dataset has a strong cohort structure that constrains interpretation.** Peptides of
   length 15 are 58% of the rows and carry 202 of the 232 failures — 87%. Failure rate is
   3.6% at lengths 8–14, **19.5% at length 15**, and 6.9% at lengths 20–25. The relationship
   is not monotone, so it is not a chain-length effect.

3. **Mean hydrophobicity is the signal that survives stratification.** Within the homogeneous
   length-15 stratum, mean Janin hydrophobicity reaches **AUC 0.740** — marginally higher
   than on the full dataset (0.734). Chain length contributes nothing within that stratum,
   by construction.

4. **Localised hydrophobicity outperforms whole-sequence composition.** A maximum 5-residue
   Kyte-Doolittle window reaches AUC 0.715 and is selected third by every statistical filter,
   while compositional counts of aromatic and polar residues are indistinguishable from
   chance. Position matters more than totals.

5. **The best model is at the published ceiling, not past it.** Random Forest on the full
   feature set reaches AUC 0.782 overall and 0.753 within-stratum, against a reported best
   of 0.773. No configuration among 46 tested exceeds it meaningfully, which is the expected
   outcome for an honest pipeline on this data.

6. **Several well-established chemical liabilities are invisible to this label.** The
   Asp-Gly aspartimide motif scores AUC 0.504. This is a property of the measurement, not of
   the chemistry (§6).

7. **The earlier feature table in this repository was mislabelled** and has been rebuilt;
   all results here use the corrected table (§7).

---

## 2. What is actually being predicted

This is the single most important constraint on every claim below, and it is easy to state
too loosely.

From the source supporting information, verbatim:

> We calculated the 'Abundance Ratio' as the sum of the relative abundance values of the m/z
> peaks matching the ordered peptide divided by the sum of the relative abundance values of
> all m/z peaks observed in the MS1 spectrum. We considered MS1 spectra with an Abundance
> Ratio < 50% as indicative of problematic (or failed) syntheses (label = 0) and MS1 spectra
> with an Abundance Ratio > 50% as successful (label = 1).

Three consequences:

**The target is crude-mixture MS1 purity, not synthesisability.** A peptide labelled 0 was
still synthesised; it came off the resin impure. Throughout this report "failure" means
*problematic synthesis / low crude purity*, never *cannot be made*.

**The threshold is arbitrary and discards information.** A peptide at 49% abundance ratio and
one at 51% are placed in opposite classes. This puts an irreducible noise floor under any
classifier. If the continuous ratio were available, regression with a rank-correlation
metric would be better posed.

**The measurement is sequence-biased.** MS1 relative abundance is not a quantitative measure
of molar purity: ionisation efficiency in positive-mode ESI varies with sequence, and basic
residues (H, K, R) ionise readily. Composition therefore influences the *measurement* as
well as the *chemistry*. This is not hypothetical — see §6.

### Dataset

| Property | Value |
|---|---|
| Peptides | 1771 |
| Label 1 (successful) | 1539 (86.9%) |
| Label 0 (problematic) | 232 (13.1%) |
| Lengths | 8–25, median 15 |
| Exact duplicate sequences | 0 |
| Near-duplicate pairs (Hamming ≤ 2) | 41, involving 82 sequences (4.6%) |

The duplicate check matters because near-identical sequences straddling a train/test boundary
inflate scores even under correct stratified cross-validation. At 4.6% this dataset is clean
on that axis, so no sequence-level grouping was required in the splits.

---

## 3. Features

Three families, all computable from sequence before synthesis.

**Physicochemical** (15) — Cruciani properties, instability index, Boman index, hydrophobic
moment, aliphatic index, isoelectric point, net charge, and five hydrophobicity scales
(Janin, Engelman, Kyte-Doolittle, Hopp-Woods, Eisenberg, Roseman), from `peptides` 0.5.0.

**Compositional** (7) — residue-group counts using the *source paper's* groupings, so results
are directly comparable to its published benchmark: `h` (I,L,M,V), `s` (A,G,P), `l` (N,Q),
`r` (F,W,Y), `p` (C,S,T), `b` (H,K,R), `a` (D,E). A standard biochemical convention is also
implemented; the two are **not** interchangeable and are never mixed.

**Positional and motif** (20) — the extension contributed by this work. Whole-sequence means
and totals discard position, and position is what drives on-resin aggregation: `ILVMAAAAAA`
and `AAILAAVMAA` have identical composition and identical mean hydrophobicity but behave
differently during chain assembly. Added:

| Feature | Rationale |
|---|---|
| `h_stretch` | Longest run of I/L/M/V — the source paper's own `h stretch` |
| `max_kd_window5` | Maximum 5-residue mean Kyte-Doolittle: localised, not global, hydrophobicity |
| `max_pbeta_window6` | Windowed Chou-Fasman β-sheet propensity, independent of hydrophobicity |
| `homopolymer_run` | Longest run of one residue — backbone regularity |
| `no_disruptor_run` | Longest stretch containing neither Gly nor Pro, the two β-sheet disruptors |
| `anchor_zone_frac` | Hydrophobic content 8–25 residues from the C-terminal anchor |
| `aspartimide_score`, `motif_DG`, `motif_nterm_Q`, `motif_cterm_P`, … | Named side-reaction motifs |

The mechanistic basis is Houben-Weyl E22a §1.2.1.3, which identifies sequence-dependent
association of protected peptide chains into β-sheet-like structures — rendering the reactive
termini inaccessible — as the dominant cause of incomplete coupling. That source also reports
the canonical difficult sequence as **decaalanine**, which contains no hydrophobic residue in
the usual sets; hence `homopolymer_run` and `no_disruptor_run`, which encode backbone
regularity rather than side-chain hydrophobicity.

---

## 4. Method

- **Cross-validation.** Repeated stratified k-fold. Everything that learns from data —
  scaling, resampling, feature selection, the model — sits **inside** the fold. Nothing is
  fitted on the full dataset.
- **Class imbalance.** Two treatments reported: `class_weight='balanced'` (§5.3) and SMOTE
  applied per-fold to training data only (§5.4). No resampled dataset is persisted to disk.
- **Metrics.** MCC, AUC-ROC and PR-AUC, **with failure as the positive class**. Accuracy and
  positive-class F1 are deliberately excluded: at 13.1% minority a constant classifier scores
  accuracy 0.869 and F1 0.930 with MCC 0.000, so those metrics reward predicting nothing.
- **Stratification.** Every result is reported both on the full dataset and within the
  length-15 stratum, for the reason in §5.2.

---

## 5. Results

### 5.1 Reproduction of the published benchmark

Two of the source paper's single-feature AUCs, recomputed independently from its groupings:

| Feature | This work | Reported |
|---|---|---|
| `r` — aromatic count (F,W,Y) | 0.483 | 0.487 |
| `p` — polar count (C,S,T) | 0.502 | 0.497 |

Both are indistinguishable from chance in both analyses. The agreement establishes that the
feature computation here matches the source.

### 5.2 The length cohort

| Length band | n | failures | failure rate |
|---|---|---|---|
| 8–14 | 644 | 23 | **3.6%** |
| **15** | **1036** | **202** | **19.5%** |
| 16–19 | 33 | 3 | 9.1% |
| 20–25 | 58 | 4 | 6.9% |

Length-15 peptides are 58.5% of the dataset and carry 87% of all failures, at 5.4× the
failure rate of the 8–14 band. The 20–25 band, which a pure chain-length effect would predict
to be worst, is near the best.

This is the signature of a **cohort effect**: the length-15 peptides are evidently a distinct
sub-library, and whatever differs about them dominates the label. Length consequently acts as
a near-perfect indicator for that cohort.

> **This is a property of the dataset, not an error by its authors.** It does, however,
> constrain what can be concluded from length, and any analysis of this dataset that reports
> length as a predictor should report the stratified figure alongside it.

### 5.3 What survives stratification

AUC for predicting failure, computed on the full dataset and within the length-15 stratum
(n = 1036, 202 failures), where no cohort indicator is available to any model:

| Feature | AUC (all) | AUC (len = 15) | Verdict |
|---|---|---|---|
| `hydrophobic_janin` | 0.734 | **0.740** | **Real.** Strongest signal; slightly stronger in-stratum |
| `hydrophobic_kytedoolittle` | 0.717 | 0.727 | Real, equivalent to Janin |
| `max_kd_window5` | 0.715 | 0.697 | Real; localised hydrophobicity |
| `n_paper_h` — I/L/M/V count | 0.645 | 0.608 | Real, modest |
| `homopolymer_run` | 0.614 | 0.606 | Real; **beats the paper's own `h stretch`** |
| `max_pbeta_window6` | 0.620 | 0.599 | Real, modest |
| `h_stretch` | 0.606 | 0.581 | Weaker than the mean-based features |
| `no_disruptor_run` | 0.636 | 0.577 | Partly length-driven |
| `aspartimide_score` | 0.533 | 0.524 | No signal (§6) |
| `n_paper_p` — polar | 0.522 | 0.503 | No signal |
| `motif_DG` | 0.504 | 0.500 | No signal (§6) |
| **`peptide_len`** | **0.639** | **0.500** | **Carries no within-cohort information** |
| `n_paper_r` — aromatic | 0.483 | 0.482 | No signal |
| **`n_paper_b` — basic** | **0.322** | **0.279** | **Strongly inverted** (§6) |

**On length, stated precisely.** Its AUC of 0.500 within the stratum is a mathematical
consequence of length being constant there, not evidence that length is irrelevant to peptide
synthesis. Chain length is a real and fundamental constraint — stepwise yield is
multiplicative, so a 30-mer at 99% per cycle retains ~74% and at 95% retains ~21%. The
correct statement is narrower and still useful:

> In this dataset, length's apparent predictive power is substantially attributable to cohort
> structure, and it carries no information within a homogeneous cohort. Mean hydrophobicity
> does.

It is also worth separating two mechanisms that length conflates. **Cumulative yield loss** is
monotone in length. **Aggregation** is not: Houben-Weyl reports its onset as a window roughly
8–10 residues out from the resin, and states that synthesis "will rarely collapse after 25
residues." These are different phenomena and should not share a single feature.

### 5.4 Model performance

Out-of-fold, 5-fold stratified × 5 repeats, `class_weight='balanced'`, failure as positive:

| Feature set | Model | MCC | AUC | PR-AUC | Recall (failures) |
|---|---|---|---|---|---|
| baseline: length + Janin | LogReg | 0.265 | 0.762 | 0.320 | **0.712** |
| named (rebuilt) | LogReg | 0.275 | 0.770 | 0.363 | 0.675 |
| named | RandForest | 0.332 | 0.765 | 0.411 | 0.348 |
| named + X4/X5 | RandForest | 0.337 | 0.773 | 0.430 | 0.340 |
| **named + X4/X5/X8** | **RandForest** | **0.355** | **0.782** | **0.447** | 0.351 |
| named + X4/X5/X8 | LogReg | 0.204 | 0.685 | 0.252 | 0.403 |

Within the length-15 stratum (19.5% failure rate), the same ordering holds, with the best
model at MCC 0.366 / AUC 0.753 / PR-AUC 0.505.

**Three observations.**

*The ceiling is real.* Best AUC here is 0.782 against a published best of 0.773. The
complexity of a 482-feature Random Forest buys about 0.02 AUC over a two-feature logistic
regression. Any result on this dataset materially above ~0.85 should be treated as a leakage
signal rather than a breakthrough.

*The 400 dipeptide features are not noise, but they are model-dependent.* A permutation test
on the best of 400 gives p = 0.005 (observed |AUC − 0.5| = 0.076 against a null ceiling of
0.038). They improve Random Forest and badly damage Logistic Regression (MCC 0.303 → 0.204),
because 400 sparse collinear columns overwhelm a linear model. Their top ten — `AA`, `VA`,
`LV`, `AV`, `VV`, `AI`, `IV`, `FA`, `FF`, `GA` — are all pairs of small or
aliphatic-hydrophobic residues: **the aggregation signal fragmented into two-residue
windows.** The named positional features recover roughly three times the per-feature signal
by aggregating over a longer, mechanistically chosen span.

*Failure recall is where models diverge, and it is the number the application needs.* The
question a chemist asks is "which of these will give me trouble?" Logistic regression with
balanced class weights recovers **71%** of failures; Random Forest recovers **35%** while
scoring better on MCC and PR-AUC. That is a threshold and cost trade-off to be chosen
deliberately, not a defect — but it must be stated rather than hidden behind a single
headline metric.

> **PR-AUC must not be compared across strata.** Chance equals the failure rate, not 0.5, so
> the stratum's higher absolute PR-AUC (0.505 vs 0.447) mostly reflects its higher base rate.
> Relative to chance: 3.4× on the full set, 2.6× in-stratum.

### 5.5 Feature-selection comparison

46 classifier × selector combinations, 10-fold × 3 repeats, SMOTE applied per-fold to
training data only.

| Rank | Method | MCC | AUC |
|---|---|---|---|
| 1 | Random Forest + chi-square (in-CV) | 0.311 | 0.751 |
| 2 | SelectFromModel (RF selector) + RF | 0.309 | 0.752 |
| 3 | SelectFromModel (RF selector) + LogReg | 0.298 | 0.758 |
| 4 | Random Forest + KS (in-CV) | 0.293 | 0.746 |
| 5 | Logistic Regression + chi-square (in-CV) | 0.292 | 0.750 |
| … | | | |
| 46 | Naive Bayes, all features | 0.147 | 0.644 |

The spread across all 46 is MCC 0.147–0.311 and AUC 0.644–**0.763**. **No configuration
exceeds the published ceiling**, and the gap between the best selector and simply using all
features with a Random Forest (MCC 0.289) is small.

Worth noting against the ranking above: the **highest AUC of all 46 configurations belongs to
plain logistic regression on all features** (AUC 0.763, MCC 0.266) — no feature selection at
all. Chi-square selection with a Random Forest wins on MCC because it trades AUC for a better
operating point, not because it finds a better feature subset. On this dataset, feature
selection buys interpretability and stability rather than discriminative power.

Features selected most consistently across folds, in rank order:

```
1 hydrophobic_janin        6 cruciani_prp1        11 n_paper_h
2 hydrophobic_kytedoolittle 7 hydrophobic_engelman 12 charge
3 max_kd_window5            8 boman               13 hydrophobic_roseman
4 hydrophobic_eisenberg     9 peptide_len         14 no_disruptor_run
5 n_paper_b                10 isoelectric_point   15 aliphatic_index
```

Hydrophobicity scales occupy four of the top eight. The windowed hydrophobicity feature
introduced here ranks third. Length ranks only ninth.

**Two cautions on reading this ranking.** `n_paper_b` at rank 5 is the basic-residue count,
which carries the ionisation confound of §6 — its high rank may be a detector effect rather
than chemistry. And `peptide_len` at rank 9 is partly a cohort indicator, per §5.2.

---

## 6. What the outcome measure cannot see

Two results are better explained by the measurement than by the chemistry, and both were
predicted in advance from the label definition in §2.

**The aspartimide motif has no signal.** Asp-Gly is the highest-sensitivity motif for
aspartimide formation in the vendor and reference literature, and base-mediated aspartimide
formation is among the best-documented side reactions in Fmoc SPPS. Here `motif_DG` scores
AUC 0.504 — chance. The ordered explanations are: (i) the assay cannot resolve it, (ii) the
protocol already suppressed it, (iii) it is rare in this sequence space. Explanation (i) is
sufficient: several aspartimide by-products are isobaric or near-isobaric with the target, so
an MS1 abundance-ratio measure largely cannot distinguish them. **The absence of signal is
evidence about the assay, not a refutation of the chemistry.**

**The basic-residue count runs backwards.** `n_paper_b` scores AUC 0.322 overall and **0.279
within-stratum** — that is, basic content strongly predicts *success*. By magnitude
(|AUC − 0.5| = 0.221) it is the second most predictive feature in the dataset, behind only
mean Janin hydrophobicity. The source paper likewise reports `length + b` among its strongest
pairs.

There is a mechanistic reading — basic residues improve solvation of the growing chain — but
there is also a measurement reading: H, K and R ionise readily in positive-mode ESI, raising
the apparent abundance ratio and hence the label. **These cannot be separated with this
dataset**, and the feature should not be given a causal interpretation without an orthogonal
purity measure such as HPLC with UV detection.

Also invisible to this label by construction: racemisation, pyroglutamate formation,
diketopiperazine chain loss, oxidation, and all cleavage and side-chain-deprotection losses.

---

## 7. Data integrity

The feature table used in earlier runs of this project was defective, and all results above
use a rebuilt table.

**The fault.** The generating code computed a `[count, fraction]` pair for each of nine
residue groups — 18 values — and wrote only the **first nine** into nine differently-named
columns. Eight of the nine therefore held a different quantity than their name stated
(`small_group` held the *fraction of Tiny*, `aliphatic_group` the *count of Small*, and so
on), and the polar, charged, basic and acidic counts were absent from the dataset entirely.
Every feature-importance interpretation produced from that table named the wrong feature.

**Verification.** Rebuilding every historical value from the source spreadsheet under that
hypothesis reproduces **15,939 of 15,939** compositional cells and **46,046 of 46,046** cells
overall, with zero mismatches. The reconstruction is retained as a regression test
(`--legacy-check`).

A second, independent defect was found in the same work: the generating script's Aliphatic
group was changed to include methionine without regenerating the data, so the committed
script and the committed data disagreed on the 468 rows containing M.

Earlier result sets are classified by validity in [`../results/README.md`](../results/README.md).

---

## 8. Limitations

1. **Single dataset, single source.** All conclusions are about one published library
   measured by one method in one laboratory. Provenance beyond the source paper is not
   recorded, and the interlaboratory literature identifies laboratory as a dominant stratum
   in peptide-synthesis outcomes.
2. **Label is a thresholded proxy.** See §2. Crude MS1 purity cut at 50%, not synthesisability.
3. **Length range is narrow and unbalanced.** 8–25 residues with 58% at length 15. Therapeutic
   peptides are routinely 30–40 residues; nothing here extrapolates there.
4. **Cohort structure cannot be removed, only stratified around.** The length-15 sub-library
   differs from the rest in ways the dataset does not record.
5. **Process conditions are unrecorded.** Resin, loading, coupling reagent, base, solvent,
   temperature and deprotection regime are all absent, so no process factor can be assessed
   and sequence-level conclusions are implicitly conditioned on whatever protocol was used.
6. **Effect sizes are dataset-specific.** The AUCs and lifts here should be recalibrated
   before being applied to any other peptide set.

---

## 9. Conclusions

The defensible claim from this work:

> **Within a homogeneous cohort of 1036 peptides, mean hydrophobicity predicts low crude
> synthesis purity at AUC 0.740, and localised windowed hydrophobicity at 0.697 — consistent
> with the β-sheet aggregation mechanism documented in the primary reference literature.
> Chain length carries no within-cohort information in this dataset, and compositional counts
> of aromatic and polar residues are indistinguishable from chance. The best model tested,
> among 46 configurations, reaches AUC 0.782, at the published ceiling of 0.773 rather than
> beyond it.**

The methodological contribution is that **position matters more than composition**: windowed
and run-length features derived from the aggregation mechanism outperform the whole-sequence
counts that dominate the existing feature set, and one of them — a maximum 5-residue
hydrophobicity window — ranks third among all features by consistency of selection.

The cautionary contribution is that **two of the strongest apparent signals in this dataset
are confounded**: length with sub-library membership, and basic-residue content with
mass-spectrometric response. Both are reproducible from the published supporting information,
and both should be reported alongside any model trained on this data.

---

## 10. Reproducing

```bash
pip install -r requirements.txt

python src/build_dataset.py                  # data/processed/peptide_features.csv
python src/build_dataset.py --legacy-check   # 46,046 cells, 0 mismatches vs the old table
python src/evaluate.py                       # §5.3, §5.4 -> results/rebuilt/
CV_REPEATS=3 python src/feature_selection.py # §5.5 -> results/smote_in_cv/<timestamp>/
```

Supporting documentation: [`01-reference-landscape.md`](01-reference-landscape.md) (prior
art), [`02-source-dataset-and-label.md`](02-source-dataset-and-label.md) (label definition and
cohort analysis), [`03-dataset-audit-findings.md`](03-dataset-audit-findings.md) (data
integrity), [`04-houben-weyl-review.md`](04-houben-weyl-review.md) (mechanistic literature
review).
