# UOOP Project — Documentation

Reference material and findings for the peptide synthesis-success prediction project.

| Doc | Contents |
|---|---|
| [01-reference-landscape.md](01-reference-landscape.md) | External tools and prior art (Peptide Companion, PepFuNN, DLOmix, Peptimizer, peptide.bio) |
| [02-source-dataset-and-label.md](02-source-dataset-and-label.md) | Label definition, the published AUC ceiling, and **§5: the length-15 cohort confound** |
| [03-dataset-audit-findings.md](03-dataset-audit-findings.md) | **Verified defects in this repo's data and pipeline.** Read before trusting any result in `results/` |
| [04-houben-weyl-review.md](04-houben-weyl-review.md) | Review of the Houben-Weyl E22a–c reference volumes: 36 factors, and six internal contradictions in the source. **Machine-generated and only partially verified — read its provenance header before citing it** |

Primary chemistry sources are in this folder: `SPPS Tips For Success Handout.pdf` (practical
heuristics) and `Marta de Zotti - materials/HOUBEN WEYL E22A-C.pdf` (the definitive reference;
§1.2.1.3 of E22a defines "difficult sequences").

A working rule engine built from those sources lives in
`.claude/skills/spps-synthesizability-review/` — run it with:

```bash
python .claude/skills/spps-synthesizability-review/scripts/scan_peptides.py data/raw/peptide_baza.csv
```

## Quick orientation

The project predicts whether a peptide can be successfully made by Fmoc solid-phase
peptide synthesis (SPPS), from its amino-acid sequence alone.

- **Data source:** supporting information of Gutman et al. 2022, *ACS Omega* (`data/raw/ao2c02425_si_002.xlsx`)
- **Working dataset:** `data/raw/peptide_baza.csv` — 1771 peptides, lengths 8–25, 1539 success / 232 failure (13.1% minority)
- **Published ceiling on this task:** AUC ≈ 0.77. Treat any result meaningfully above that as a leakage signal, not a breakthrough.

## The three things to know

1. **The label is MS1 crude purity thresholded at 50%** — not "can this peptide be made".
   A peptide labelled 0 was still synthesised; it came off impure. ([02](02-source-dataset-and-label.md) §1)
2. **Length-15 peptides are 58% of the data and carry 87% of the failures.** Length acts as a
   cohort indicator, so the source paper's strongest single feature does not survive
   stratification. Mean hydrophobicity does, at AUC 0.740 within-cohort.
   ([02](02-source-dataset-and-label.md) §5)
3. **The compositional feature columns are mislabelled** — 8 of 9 contain a different quantity
   than their name says, which invalidates every feature-importance interpretation currently in
   `results/`. ([03](03-dataset-audit-findings.md) Finding 1)

## Status

`03-dataset-audit-findings.md` documents defects confirmed by reproducing the feature
computation against the committed CSVs, audited against `develop`. The leakage and metric
problems have been **fixed** in `src/feature_selection.py`; the mislabelled columns and the
committed credential are **open**.
