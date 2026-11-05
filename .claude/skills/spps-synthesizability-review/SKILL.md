---
name: spps-synthesizability-review
description: Use when reviewing any content - papers, patents, vendor handbooks, protocols, codebases, datasets, model cards - for what it claims about factors affecting peptide synthesis outcome, when screening sequences for synthesis liabilities, or when auditing a synthesizability model or descriptor set. Triggers include peptide, SPPS, Fmoc, Boc, solid-phase synthesis, synthesizability, synthetic accessibility, coupling efficiency, crude purity, difficult sequence, aggregation, aspartimide, racemization, deletion sequence, or amino-acid sequence data.
---

# Peptide Synthesis Factor Review

Extract, from any source, what it claims about factors affecting peptide synthesis outcome —
then judge whether the claim is supported.

## Domain context

Peptides are short amino-acid chains. Most are made by **solid-phase peptide synthesis
(SPPS)**: the chain is anchored to a resin bead and built one residue at a time, C-terminus
to N-terminus. Each cycle is (1) remove the temporary N-terminal protecting group — in Fmoc
chemistry, with piperidine; (2) couple the next activated, side-chain-protected residue.
After the last cycle the chain is cleaved from the resin and the side-chain protection
removed, usually with TFA.

Two consequences drive everything:

- **Yield is multiplicative.** At 99% per cycle a 30-mer gives ~74% crude; at 95% it gives
  ~21%. Small per-cycle losses compound into failure.
- **Failures are mostly not "the bond won't form."** The dominant cause is **loss of
  solvation**: as the chain grows, sequence-dependent intermolecular association into
  β-sheet-like structures buries the reactive termini so reagents cannot reach them. This is
  the classical "difficult sequence" phenomenon. Everything else is a discrete, nameable side
  reaction with a known mitigation.

So when a source claims some factor affects synthesis, it is almost always asserting a
mechanism in one of three places. Classify it before evaluating it.

## The three axes — what you are looking for

| Axis | What it covers | Portable across labs? |
|---|---|---|
| **Sequence-intrinsic** | Properties computable from the sequence alone: hydrophobic runs, β-branched residues, specific motifs (Asp-Xaa, C-terminal Pro, N-terminal Gln), length, charge | **Yes.** Transfers between labs and instruments. |
| **Process / environment** | Resin and linker, coupling reagent and base, solvent, temperature, excess and cycle time, deprotection regime, microwave, flow | **Partly.** Depends on equipment and protocol. |
| **Analytical / measurement** | How "success" was defined and measured: crude HPLC purity, MS peak abundance, isolated yield, resin test, and any threshold applied | **No** — but it determines what any dataset can and cannot show. |

**A claim is only as transferable as its weakest axis.** A sequence rule validated on one
instrument under one coupling protocol is a claim about that protocol too.

Full catalogue of known factors on each axis, with mechanisms and the vocabulary sources use
for them: `references/factor-catalogue.md`. Read it before reviewing unfamiliar material —
it is the lookup table for recognising a factor when a paper names it obliquely. A worked
end-to-end review, including two cases where the chemistry and the measurement disagreed:
`references/worked-example.md`.

## Workflow

**1. Scope.** Write down the question before reading. "Which factors does this source claim
affect outcome, and how well does it support each?" is a different job from "can these
specific peptides be made?"

**2. Locate the outcome definition first.** Before any other reading, find how the source
defines success. In a paper it is in Methods; in a dataset it is the label column's
provenance; in a codebase it is whatever the target column is built from. **If you cannot
find it, say so and stop treating the source's numbers as comparable to anything else.**
Everything downstream depends on this.

**3. Read for factors, not for conclusions.** Work through the source recording each asserted
factor as a finding record (below). Vendor handbooks and textbook chapters are often denser
in usable mechanism than primary papers; do not skip them because they are not peer-reviewed.

**4. Grade the evidence** on the ladder below. Record the grade with the finding.

**5. Cross-check.** For each factor, ask whether another source in your set contradicts it,
and whether the claimed mechanism and the measurement can actually see each other. A
liability that produces a by-product the assay cannot detect will show no signal in that
dataset — which is evidence about the assay, not about the chemistry.

**6. Synthesize** into the three axes, separating what is established from what one source
asserts once.

### Per-source playbooks

- **Primary paper / preprint** — Methods for the outcome definition and the synthesis
  protocol; any supplementary table of per-feature performance (this is usually where the
  honest numbers are, and it is often the only place single-feature baselines appear);
  cohort structure of the dataset. Check whether reported performance is against a stated
  baseline.
- **Vendor handbook / application note** — rich in mechanism and mitigation, and often the
  only place graded motif tables appear. Treat quantitative claims as unreplicated and
  interest-bearing: the vendor sells the fix. Extract the mechanism, flag the numbers.
- **Textbook / reference work** — best source for mechanism and for the controlled vocabulary.
  Usually no primary data. Cite for mechanism, not for effect size.
- **Codebase** — find where features are computed and where the label is constructed, then
  **recompute two or three features from raw input and compare to the stored values.** Do not
  trust a column because of its name. Check whether resampling, feature selection or scaling
  happen inside or outside the cross-validation split.
- **Dataset / model card** — label provenance, class balance, length and cohort structure,
  duplicate and near-duplicate sequences, and which metrics are reported. See Auditing below.

## Finding record

Record each asserted factor as:

```
factor        what is claimed to affect outcome
axis          sequence-intrinsic | process-environment | analytical-measurement
direction     helps | harms | non-monotone | unclear
mechanism     why, in one sentence - or "asserted, no mechanism given"
evidence      grade from the ladder below
scope         sequence lengths, chemistry, instrument the claim was established on
detectable    computable from sequence alone? needs process metadata? measured post hoc?
conflicts     other sources that disagree
```

`detectable` matters more than it looks. Features derived from observed spectra — for
instance, which residue was lost in a failed run — cannot be used to score a peptide that
has not been made yet. Sources mix these in with predictive features freely.

## Evidence ladder

| Grade | What it takes |
|---|---|
| **A** | Controlled experiment isolating the factor, replicated, with the outcome measure stated |
| **B** | Large labelled dataset with held-out validation and a stated single-feature baseline |
| **C** | Retrospective association in one dataset, no baseline or no held-out set |
| **D** | Mechanistic argument, worked examples, no systematic data — most textbook and vendor content |
| **E** | Assertion by citation chain, where the cited source does not support the specific claim |

Grade D is not worthless; most reliable SPPS mechanism sits there. The failure is citing D
as if it were A.

## Claims to challenge

| Claim | Challenge |
|---|---|
| "Feature X predicts synthesis success" | Against what baseline? What is the single best feature alone? A complex model that does not beat one or two descriptors has not earned its complexity. |
| A headline accuracy or F1 on an imbalanced set | At 10–15% minority, a constant classifier can score ~0.87 accuracy and ~0.93 F1 with MCC 0.000. Demand MCC, AUC and PR-AUC **with failure as the positive class**. |
| A strong feature that correlates with a dataset's structure | Stratify. If one length, batch or source library holds a disproportionate share of failures, a correlated feature may be a cohort indicator. Re-check the AUC within a stratum. |
| A sequence rule from mean or total descriptors | Means and counts discard position. Two sequences with identical composition and opposite clustering behave differently. Ask whether run-length or windowed features were tried. |
| Composition counts of a residue class | Composition is not mechanism. Localisation usually matters more than totals. |
| A charge or basicity effect on an MS-derived outcome | Basic residues ionise well in positive-mode ESI, inflating apparent purity. The "effect" may be in the detector. Never read it causally without an orthogonal measure. |
| "Non-synthesizable peptide" | Almost always means *low crude purity*, not *cannot be made*. Correct the language; it changes what the result means. |
| A threshold-derived binary label | Binarising a continuous measure puts near-identical samples in opposite classes. Ask whether the continuous value is available; regression on it is better posed. |

## Auditing a dataset or model

1. **Label provenance** — what physical quantity, measured how, thresholded where.
2. **Column integrity** — recompute features from raw sequences and compare. Mislabelled or
   offset descriptor columns are common and silently invalidate every importance ranking.
3. **Redundancy** — counts alongside their own fractions, with length also present, inject
   near-deterministic collinearity into feature selection.
4. **Cohort structure** — failure rate by length, by source, by batch. Look for one stratum
   dominating.
5. **Leakage** — resampling or feature selection fitted outside the CV split; a persisted
   resampled file is a strong smell. Synthetic rows generated for oversampling must keep
   their sequence and their descriptors consistent, and interpolating one-hot or count
   columns produces values that correspond to no real peptide.
6. **Near-duplicates** across splits.
7. **Metrics** — see the table above.

## Screening sequences

When the content contains sequences, run the scanner rather than reading them by eye:

```bash
python scripts/scan_peptides.py <file>            # CSV/TSV/XLSX/FASTA/text, auto-detects column
python scripts/scan_peptides.py <file> --format json|csv
```

Output splits into **chain-assembly difficulty** and **process liabilities**, because they
answer different questions: the first bears on whether the chain assembles, the second is a
protocol checklist. `--help` for all options.

### Calibration

The scanner ships **no hardcoded effect sizes**, deliberately — predictive power is a property
of a dataset and its outcome definition, not of the chemistry. If you have labelled outcomes,
derive them:

```bash
python scripts/scan_peptides.py <file> --label-column <col> [--failure-value <v>]
```

This reports per-rule lift, per-feature AUC, and automatically runs the cohort check,
re-computing AUC within a dominant stratum when it finds one. **Use these numbers for the
dataset you ran them on and nowhere else.** A rule with lift near 1.0 is not refuted as
chemistry — it may be invisible to that outcome measure.

## Tools

Use whatever this session has: a literature MCP (PubMed or similar) for primary sources and
full text; web search for vendor handbooks, application notes and supplementary material, which
are often not indexed in biomedical databases; chemistry databases for compound and drug
context. Check availability rather than assuming — servers vary per session, and a failed
connection is not an absent capability. Pure-chemistry journals are frequently outside
biomedical indexes, so search the web for those directly.

## Honest limits

The scanner emits **mechanistic flags, not calibrated probabilities**. Its thresholds are
judgement calls from mechanism, not fitted values. The evidence base in the catalogue is
weighted toward short-to-medium synthetic peptides; therapeutic peptides are routinely 30–40
residues and much of this transfers only partly. Say which axis a conclusion rests on and what
scope it was established at, rather than implying precision you do not have.
