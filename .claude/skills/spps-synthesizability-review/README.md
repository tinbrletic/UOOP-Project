# SPPS Synthesizability Review

An agent skill for finding **what affects peptide synthesis outcome in any given content** —
a paper, a patent, a vendor handbook, a lab protocol, a codebase, a dataset or a model card —
and for judging whether what it claims is actually supported.

Paired with a deterministic scanner that screens amino-acid sequences for synthesis
liabilities and calibrates its own rules against whatever labelled data you have.

---

## What problem it solves

A researcher facing a new source on peptide synthesis has three recurring difficulties:

1. **Recognising a factor when a source names it obliquely.** The same mechanism appears as
   "difficult sequence", "aggregation", "poor solvation", "incomplete coupling" and "chain
   association" across four sources.
2. **Knowing what a source's numbers can and cannot mean.** A crude HPLC area fraction, an MS1
   abundance ratio, a resin colour test and a UV deprotection-peak shape are four different
   quantities, routinely pooled as "synthesis success".
3. **Separating a real effect from an artifact of the measurement**, the dataset's cohort
   structure, or the laboratory it came from.

The skill gives an agent a domain primer, a four-axis taxonomy for classifying any claimed
factor, a workflow that puts the outcome definition first, an evidence ladder, and a
catalogue of known factors with their mechanisms and the vocabulary sources use for them.

## Install

Drop the directory into your agent's skills folder:

```bash
cp -r spps-synthesizability-review ~/.claude/skills/        # personal
# or, for one project:
cp -r spps-synthesizability-review <your-repo>/.claude/skills/
```

**Requirements.** Python **3.8+**, standard library only. `openpyxl` is needed *only* to read
`.xlsx` input; everything else works without it.

```bash
python3 -m pip install openpyxl     # optional
```

No build step, no configuration, no network access. Verified to run from a copy outside any
repository.

## Use

**As a skill.** Point an agent at content and ask what it says about factors affecting peptide
synthesis. The agent reads `SKILL.md`, loads the references it needs, and follows the
workflow. Nothing else to invoke.

**As a tool.** The scanner is useful on its own:

```bash
# screen sequences for liabilities (CSV/TSV/XLSX/FASTA/plain text, column auto-detected)
python3 scripts/scan_peptides.py demo/example.fasta

# derive effect sizes on YOUR labelled data, with an automatic cohort check
python3 scripts/scan_peptides.py mydata.csv --label-column outcome

# sanity-check the rules against known-difficult sequences and benign controls
python3 scripts/scan_peptides.py --selftest
```

`--help` for all options. See [`demo/README.md`](demo/README.md) for a five-minute walkthrough
with expected output.

## What's in here

| File | |
|---|---|
| `SKILL.md` | The method: domain primer, four axes, workflow, finding record, evidence ladder |
| `references/factor-catalogue.md` | Known factors on all four axes, with mechanisms, vocabulary, and page-level citations where they could be verified |
| `references/review-checklists.md` | Dataset/model audit (nine ordered steps); protocol for reading scanned sources without misquoting them |
| `references/validation.md` | How the skill was tested, what the tests found, **and where they were weak** |
| `references/worked-example.md` | A worked end-to-end review |
| `scripts/scan_peptides.py` | The scanner: sequence screening, calibration, self-test |
| `demo/` | Runnable walkthrough with example input |

## Design commitments

These are the choices that shaped it, and the ones worth arguing with:

**No hardcoded effect sizes.** Predictive power is a property of a dataset and its outcome
definition, not of the chemistry. The scanner ships none, and `--label-column` derives them
for the data in front of you.

**Thresholds come from mechanism, not from fitting.** A hydrophobic run long enough to
nucleate a β-sheet is a claim about chemistry, not a hyperparameter. The scanner reports a
rule as *untested* on data that cannot exercise it rather than retuning to make it fire.

**Severity tiers are measured, not asserted.** Calibration prints failure rate per tier
against the base rate. On one labelled set the MEDIUM tier came out *below* base rate — so it
means "unremarkable", not "at risk", and the default listing floor is HIGH.

**Absence of a citation is the mark.** Roughly 26 catalogue claims carry a verified page
reference; the rest carry none and say so. A plausible-looking reference to an unchecked claim
is worse than an admitted gap.

**Weak results are classified, not dismissed.** A rule showing no signal is *untested*,
*degenerate*, *invisible to the outcome measure*, or *refuted* — in that order of likelihood.
"The chemistry is wrong" comes last and rarely applies.

## Honest limits

- The scanner emits **mechanistic flags, not calibrated probabilities**.
- The evidence base skews to **short-to-medium peptides**; therapeutic peptides run 30–40
  residues and much of this transfers only partly.
- **The guidance wording has not been pressure-tested.** Whether specific phrasings bind under
  competing incentives is unmeasured.
- Validation used **n=1 per arm** in two of three rounds. See `references/validation.md`,
  which also records the flaws in each test.

## Provenance

Built from: Houben-Weyl *Methods in Organic Chemistry* Vol. E22a–c (Thieme 1999); Gyros Protein
Technologies, *SPPS Tips For Success*; Gutman et al., *Predicting the Success of Fmoc-Based
Peptide Synthesis*, ACS Omega 2022; Tamás et al., *Amino acid composition drives aggregation
during peptide synthesis*, Nat. Chem. 2026, doi:10.1038/s41557-026-02090-0.

Tested against the first of those, a public ML codebase, and the last — none of which it was
built from at the time of testing.
