# Worked example — auditing a published MS-labelled synthesis dataset

An illustration of the SKILL.md workflow end to end. **The numbers here belong to one
dataset. Do not reuse them.** What transfers is the sequence of moves.

Source: Gutman et al., *Predicting the Success of Fmoc-Based Peptide Synthesis*,
ACS Omega 2022 — a public dataset of ~1771 peptides with a binary synthesis outcome, plus
supporting information containing a per-feature performance table.

---

## Step 2 — Locate the outcome definition first

From the supporting information header, not the abstract:

> Abundance Ratio = Σ(relative abundance of m/z peaks matching the ordered peptide) ÷
> Σ(relative abundance of all m/z peaks in the MS1 spectrum). Ratio < 50% → label 0
> (problematic/failed); > 50% → label 1 (successful).

Three consequences recorded immediately, before reading anything else:

- **Axis 3 classification: MS peak abundance ratio.** Per the catalogue, this carries a
  sequence-dependent ionisation bias and cannot see isobaric by-products.
- **Thresholded at 50%** — a continuous quantity binarised; near-identical peptides split
  across classes.
- **"Failure" means low crude purity, not "cannot be made."** Every downstream statement had
  to be reworded accordingly.

This one step reframed the whole review.

## Step 3–4 — Read for factors, grade them

The supporting information contained a single-feature and pairwise AUC table — the most
useful object in the paper, and the kind of thing that lives in supplements rather than main
text. Extracted:

- Best single feature reported: peptide length, AUC ≈ 0.67
- Best pair: length + a hydrophobicity scale, AUC ≈ 0.77
- Several composition counts (aromatic, polar, acidic) at AUC 0.48–0.51 — **chance**
- One run-length feature present: longest consecutive aliphatic-hydrophobic stretch

Grade: **B** — large labelled dataset, held-out validation, and a stated single-feature
baseline. The baseline is what makes it B rather than C, and it is also what lets any later
model be judged.

**Ceiling established: ~0.77 AUC.** Anything materially above that on this data is a leakage
signal, not a result.

## Step 5 — Cross-check mechanism against measurement

Running the calibration mode over the labelled data:

```bash
python scripts/scan_peptides.py <data.csv> --label-column <label>
```

Two results worth the whole exercise:

**(a) A textbook liability with no signal.** The Asp-Gly aspartimide motif — the
highest-sensitivity motif in the vendor table — scored AUC ≈ 0.50 as a standalone feature.
Not a refutation of the chemistry. Per the catalogue's three ordered explanations: aspartimide
by-products are often isobaric or near-isobaric with the target, so an MS1 abundance ratio
largely cannot see them. **This is evidence about the assay.**

**(b) A rule running backwards.** The Arg rule fired *more* often on successes (lift ≈ 0.58).
The catalogue predicts exactly this: basic residues ionise well in positive-mode ESI and
inflate the abundance ratio. So Arg content on this label is partly a detector effect.
Recorded as a measurement confound, and the rule demoted to INFO with a warning rather than
deleted — the δ-lactam chemistry is real, it simply is not what this label measures.

## Step 5 continued — the cohort check

The scanner's automatic cohort check flagged it:

> WARNING: length 15 holds 87% of all failures and 58% of all rows.

Failure rate was ~3.6% at lengths 8–14, **19.5% at length 15**, and ~7% at 20–25 — not
monotone, so not a chain-length effect. One length value dominating both rows and failures is
the signature of a distinct sub-library.

Re-checking within that single stratum, where no cohort indicator is available to any model:

| Feature | AUC overall | AUC within stratum |
|---|---|---|
| length | 0.639 | **0.500** |
| mean Janin hydrophobicity | 0.734 | **0.740** |
| mean Kyte-Doolittle | 0.717 | 0.727 |
| longest I/L/M/V run | 0.606 | 0.581 |

**Conclusion.** The hydrophobicity signal is real — it holds, marginally stronger, inside a
homogeneous cohort, consistent with the aggregation mechanism. The length signal is not: it
collapses to chance in-stratum. So the paper's strongest reported single feature is
substantially a cohort indicator, and its best pair is best read as hydrophobicity doing the
work alongside a batch label.

Note also that the mechanistically most satisfying feature — the run-length stretch — was
*weaker* than a plain whole-sequence mean here. Mechanism tells you where to look; it does
not tell you what will win.

## What generalises

1. **Find the outcome definition before anything else.** It reframed every later step.
2. **Supplementary tables carry the honest baselines.** Look there first.
3. **Absence of signal for a well-established liability is information about the assay**, not
   a refutation of the chemistry.
4. **A rule running backwards points at a measurement confound** — and the catalogue predicted
   which one.
5. **Stratify before believing any feature**, especially one correlated with dataset
   structure. The cohort check is automatic for a reason.
6. **Report what the outcome measure cannot see.** It is usually the most load-bearing
   sentence in the review.
