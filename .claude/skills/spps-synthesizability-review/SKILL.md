---
name: spps-synthesizability-review
description: Use when a file contains peptide sequences and the question is whether they can be made, or when auditing a peptide dataset, descriptor set, or synthesis-success model. Triggers include peptide, SPPS, Fmoc, solid-phase synthesis, synthesizability, synthetic accessibility, crude purity, aggregation, difficult sequence, aspartimide, racemization, peptide descriptors, or a CSV/FASTA/XLSX column of amino-acid sequences.
---

# SPPS Synthesizability Review

Peptides are made on solid support by Fmoc SPPS: the chain is assembled C-to-N, one residue
at a time, each cycle being Fmoc removal with piperidine then coupling of the next activated
residue. Yield is multiplicative across cycles, so small per-cycle losses compound, and some
sequences fail for reasons that are predictable from sequence alone.

**Core principle:** the dominant failure mode is not a bond that will not form — it is
**on-resin aggregation**. Hydrophobic stretches drive the growing chains into β-sheet-like
association, which buries the reactive termini and starves them of reagent. Everything else
is a discrete, nameable side reaction with a known mitigation.

## Workflow

1. **Run the scanner.** Do not eyeball sequences.
   ```bash
   python scripts/scan_peptides.py <file> [--column peptide_seq] [--format text|json|csv]
   ```
   It auto-detects the sequence column in CSV/TSV/XLSX/FASTA/text. `--help` for options.

2. **Read the two categories separately — they answer different questions.**
   - **CHAIN-ASSEMBLY DIFFICULTY** — predicts whether the chain assembles cleanly. This is
     the risk signal. Act on it.
   - **PROCESS LIABILITIES** — real chemistry needing a specific mitigation, but *not*
     predictive of crude purity. This is a protocol checklist, not a risk score.

3. **Check the label before believing any model.** If the file is a dataset with an outcome
   column, find out what that outcome physically measures before interpreting anything. See
   Confounds.

4. **If the file is a descriptor set, verify the columns contain what their names claim.**
   Recompute two or three features from the sequences and compare. Do not assume.

5. **Report findings ranked by severity, naming positions.** "Hydrophobic run of 5 at
   position 4" is actionable; "this peptide is difficult" is not.

## What actually predicts failure

Measured on the Gutman 2022 dataset (n=1771, 13.1% failure, labels from MS1 abundance ratio):

| Signal | AUC (failure) | Verdict |
|---|---|---|
| Mean Janin hydrophobicity | **0.734** (0.740 within-cohort) | **Real. Primary signal.** |
| Mean Kyte-Doolittle | 0.717 (0.727 within-cohort) | Real, equivalent to Janin |
| Max 5-residue hydrophobic window | 0.715 (0.697 within-cohort) | Real, adds position info |
| Longest I/L/M/V run (`h stretch`) | 0.606 (0.581 within-cohort) | Weak — mechanism is right, mean captures it better |
| Peptide length | 0.639 (**0.500 within-cohort**) | **Confounded. Do not trust.** |
| Asp-Gly motif present | 0.505 | No signal on this label |

**Ceiling for this task is AUC ≈ 0.77.** Treat anything above ~0.85 as leakage until proven.

## Confounds — check these before trusting a result

**Cohort confound.** In this dataset, length-15 peptides are 58% of rows and carry 87% of
all failures (19.5% failure vs 3.6% for lengths 8–14). Length therefore acts as a batch
indicator, not a chain-length effect. **Always stratify by any variable that splits the
dataset into unequal-failure-rate populations, and re-check whether your signal survives
within a stratum.** Hydrophobicity did. Length did not.

**Measurement confound.** When the label is derived from mass-spectrometry peak abundance,
the label is sequence-biased: basic residues (H, K, R) ionise well in positive ESI, raising
apparent purity. Empirically, the Arg rule fires **more** often on successes (lift 0.58).
So a model "learning" that basic residues help may be learning ionisation efficiency, not
chemistry. Never give a causal reading to charge or basicity features on an MS-derived label.

**Thresholded label.** A continuous quantity cut at 50% puts near-identical peptides in
opposite classes. If the continuous value is recoverable, regress on it and use rank
correlation.

## Common mistakes

| Mistake | Why it is wrong |
|---|---|
| Reporting accuracy or positive-class F1 | At 13% minority, a constant classifier scores accuracy 0.869 and F1 0.930 with MCC 0.000. Use MCC, AUC, and PR-AUC with **failure as the positive class**. |
| Using only mean/total descriptors | `ILVMAAAAAA` and `AAILAAVMAA` get identical means and counts but behave differently. Means discard the position information that drives aggregation. |
| Treating 400 anonymous dipeptide counts as features | Name the mechanistically meaningful ones (e.g. Asp-Gly) instead of burying them among 400 columns at n<2000. |
| Trusting a column because of its name | Recompute and compare. Mislabelled descriptor columns are common and silently invalidate every importance ranking. |
| Persisting a resampled dataset | Resampling before splitting leaks across folds. Resample inside the CV pipeline only. |
| Calling a flagged peptide "non-synthesizable" | These labels usually mean *low crude purity*, not *cannot be made*. Say "problematic synthesis". |

## Honest limits

The scanner emits **mechanistic risk flags, not calibrated probabilities**. Lift values in
its output were measured on one dataset of mostly 14–15-mers and will not transfer unchanged
to other peptide classes or lengths. Therapeutic peptides are routinely 30–40 residues; this
evidence base does not extend there. Say so rather than implying precision you do not have.

Rule-by-rule sources and the full validation table: `references/liability-rules.md`.
