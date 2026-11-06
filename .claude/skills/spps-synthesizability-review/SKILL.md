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

**A correction to the field's own vocabulary, which you will have to work against.** The
language is coupling-centric — "coupling efficiency", "difficult *coupling*", per-coupling
models. But modern coupling reagents are efficient enough that acylation failure is largely
confined to extremely hindered amines, and the reference literature attributes deletion
sequences in Fmoc SPPS mainly to **incomplete Fmoc removal**: aggregation hinders diffusion
in both directions. One large interlaboratory study went further and found post-assembly
steps (cleavage and deprotection) more significant than assembly itself. So when a source
says "coupling", check which step it measured — and be suspicious of a model built only on
assembly-step features.

So when a source claims some factor affects synthesis, it is asserting a mechanism in one of
four places. Classify it before evaluating it.

## The four axes — what you are looking for

| Axis | What it covers | Portable across labs? |
|---|---|---|
| **Sequence-intrinsic** | Properties computable from the sequence alone: hydrophobic and homopolymer runs, β-branched residues, specific motifs (Asp-Xaa, C-terminal Pro, N-terminal Gln), position relative to each terminus, charge | **Yes.** Transfers between labs and instruments. |
| **Process / environment** | Resin, loading and linker, coupling reagent and base, solvent, temperature, excess and cycle time, deprotection regime, capping, cleavage cocktail, microwave, flow | **Partly.** Depends on equipment and protocol. |
| **Analytical / measurement** | How "success" was defined and measured: crude HPLC purity, MS peak abundance, isolated yield, resin colour test, and any threshold applied | **No** — but it determines what any dataset can and cannot show. |
| **Provenance / operator** | How many laboratories or operators contributed; whether characterisation was central or per-site; whether provenance is recorded at all | **No.** Often the largest effect, and usually unrecorded. |

**A claim is only as transferable as its weakest axis.** A sequence rule validated on one
instrument under one coupling protocol is a claim about that protocol too.

The fourth axis is there because the strongest interlaboratory evidence in this literature
concludes that **peptide quality depends as much or more on the skill of the personnel than
on the instrument or method**, and in one round-robin the high racemization values all came
from a single laboratory. That is a cohort effect with a named stratum: a dataset pooled
across facilities carries a laboratory effect big enough to dominate any sequence feature.
It fits none of the first three axes, so without this one it goes unrecorded.

Three references, loaded as needed:

| File | Use it for |
|---|---|
| `references/factor-catalogue.md` | Every known factor on all four axes, with mechanism, the vocabulary sources use for it, and whether it is detectable before synthesis. **Read before reviewing unfamiliar material** — it is the lookup table for recognising a factor when a source names it obliquely. |
| `references/review-checklists.md` | The dataset/model audit (nine ordered steps) and the protocol for reading scanned sources without misquoting them. |
| `references/worked-example.md` | A worked end-to-end review, including two cases where the chemistry and the measurement disagreed. |

## Workflow

**1. Scope.** Write down the question before reading. "Which factors does this source claim
affect outcome, and how well does it support each?" is a different job from "can these
specific peptides be made?"

**2. Locate the outcome definition for each distinct body of results — before reading for
conclusions.** A source may have several, and a large or multi-author one usually does.
Tabulate them and **never pool across them**: a resin colour-test "complete coupling", a
crude HPLC area fraction at 210 nm and a chiral-HPLC half-life are three different
quantities.

Where to look: in a paper, Methods; in a dataset, the label column's provenance; in a
codebase, whatever the target column is built from. Sometimes it must be **assembled** —
the detection wavelength that defines a purity measure can sit in a figure caption while the
protocol sits in the text. If a body of results has no locatable definition, record that and
treat its numbers as incomparable to everything else, rather than discarding the source.

This step changes the review more than any other. It is where you discover that an outcome
measure is blind to the very thing a chapter claims to demonstrate.

**3. Read for factors, not for conclusions.** Work through the source recording each asserted
factor as a finding record (below). Vendor handbooks and textbook chapters are often denser
in usable mechanism than primary papers; do not skip them because they are not peer-reviewed.

**4. Grade the evidence** on the ladder below. Record the grade with the finding.

**5. Cross-check — including against the source itself.** For each factor ask three things:
does another source contradict it; **does this source contradict itself**; and can the
claimed mechanism and the measurement actually see each other?

The self-contradiction check is the highest-yield step in the whole workflow. In a
multi-author work, **expect the mechanism chapters and the data chapters to disagree, and
prefer the measurement.** Contradictions between sections of one source are usually more
informative than contradictions between sources, because both sides were written by people
with access to the same field.

On the third question: a liability that produces a by-product the assay cannot detect will
show no signal in that dataset — which is evidence about the assay, not about the chemistry.
The ordered explanations for a well-established liability showing no signal are (1) the assay
cannot resolve it, (2) the protocol already suppressed it, (3) it is genuinely rare in that
sequence space. "The chemistry is wrong" comes last and is rarely the answer.

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
- **Single-author textbook** — best source for mechanism and for the controlled vocabulary.
  Usually no primary data. Cite for mechanism, not for effect size.
- **Multi-author reference work, handbook or edited volume — grade each chapter separately.**
  A work like this is not one source; it is dozens bound together, each with its own authors,
  its own outcome measure and its own evidence class. Chapters in a single such work can run
  from pure mechanism with no data (grade D) to controlled kinetics and blinded
  interlaboratory studies (grade A). **Grading the volume as a unit will throw away its best
  evidence**, and applying "textbook → cite for mechanism, not effect size" to the whole
  thing is a category error. Two further expectations: the mechanism chapters and the data
  chapters will contradict each other, and **the primary data are often in the analytical or
  QC chapters, not the synthesis chapters** — so do not deprioritise those when reading for
  synthesis factors.
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
axis          sequence-intrinsic | process-environment | analytical-measurement | provenance
direction     helps | harms | non-monotone | unclear  -- PER AXIS, not one verdict
tradeoff      the countervailing finding, if this factor helps on one axis and harms
              on another; else "none known"
mechanism     why, in one sentence - or "asserted, no mechanism given"
evidence      grade from the ladder below, for THIS CHAPTER, not the whole work
scope         sequence lengths, chemistry, phase, instrument, era the claim was
              established on
detectable    computable from sequence alone? needs process metadata? measured post hoc?
conflicts     other sources OR OTHER SECTIONS OF THIS SOURCE that disagree
reconstruction  "verbatim", or "reassembled from OCR fragments - verified by <method>"
```

**`direction` is per axis, and `tradeoff` exists because one verdict is usually wrong.** The
normal shape of an SPPS design decision is a residue that helps one thing and harms another
at the same position: Pro, Arg, His, Gln and Asn are all reported as *protective against
assembly failure* while Arg (δ-lactam, des-Arg deletions), His (epimerisation) and Gln
(pyroglutamate, dehydration) are *also* named liabilities. Splitting that into two records
loses the fact that they trade off at the same position.

**`detectable` matters more than it looks.** Features derived from observed spectra — which
residue was lost in a failed run, say — cannot be used to score a peptide that has not been
made yet. Sources mix these in with predictive features freely.

**`evidence` grades the chapter, never the volume.** See the multi-author playbook above.

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
| "Non-synthesizable peptide" | **Check the outcome measure before correcting this.** Where the outcome is a purity measure it means *low crude purity*, not *cannot be made*, and the language should be fixed. But diketopiperazine cleaves the chain off the resin, and pyroglutamate and trifluoroacetylation cap it permanently — those genuinely are cannot-be-made. Condition the wording on the measure; do not apply a blanket rewrite. |
| A threshold-derived binary label | Binarising a continuous measure puts near-identical samples in opposite classes. Ask whether the continuous value is available; regression on it is better posed. |
| **A reported null** | Do not skip past it. A null on a factor the field treats as a standard lever is worth more than most positive findings — one 33-laboratory study found **no correlation between coupling time and percent correct product**. Record nulls explicitly, with their power: a null from 33 labs is evidence, a null from n=1 is not. |
| A mitigation applied without a stated cost | Every mitigation has one. Backbone protection converts an aggregation problem into a steric coupling problem one position later; capping improves purifiability, not yield. An author who applied a mitigation and reported no cost either did not look or did not say. |

## Auditing a dataset or model

Nine ordered steps in `references/review-checklists.md` — label provenance, column integrity,
reproducibility, collinearity, cohort structure, leakage, near-duplicates, metrics, baseline.
**Work it in order; each step can invalidate everything after it.**

Two that catch the most: **recompute a few features from raw sequences and compare to the
stored values** rather than trusting a column's name, and **stratify by laboratory as well as
by length and batch** — provenance is the stratum the interlaboratory evidence calls dominant
and the one datasets most often fail to record.

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

### When there are no labelled outcomes

Most sources have no label column. There are still two checks worth running, and both find
real defects:

```bash
python scripts/scan_peptides.py --selftest
```

runs the rules against sequences the literature names as known-difficult, plus benign
controls. **A difficult case that draws nothing, or a control that lights up, is a rule
defect, not a result.**

Then do the same with the source in front of you: **run the scanner on the failure cases that
source names, and see whether the rules fire.** This is the nearest thing to calibration
available for a non-dataset source, it is free, and the ground truth is handed to you by the
author. It is how the poly-Ala and positional-deamidation gaps in this skill's own rule set
were found. If a source's canonical failure does not trip a rule, report that as a finding
about the rules.

## Reading scanned and OCR'd sources

Much of the best material in this field arrives as a scan, and OCR of a two-column page
**silently reorders lines** — so a quotation assembled from adjacent fragments can be wrong
while reading perfectly fluently. Scanned **table headers** are the worst case: emitted out of
order, they invert a table's meaning with no visible corruption.

Seven-step protocol in `references/review-checklists.md`. The two that matter most: **verify
every reconstructed quotation with a literal fixed-string search before using it**, and
**never trust a scanned table's header-to-column mapping on its own** — cross-check it against
the stated mechanism or numbers elsewhere in the chapter, and if they disagree, the mechanism
text wins.

## Tools

**If the source is in hand, read the source.** Local files need no external tooling, and
reaching for search when the material is already on disk is wasted effort.

Otherwise use whatever this session has: a literature MCP for primary sources and full text;
web search for vendor handbooks, application notes and supplementary material, which are
often not indexed in biomedical databases; chemistry databases for compound context. Check
availability rather than assuming — servers vary per session, and a failed connection is not
an absent capability. Pure-chemistry journals are frequently outside biomedical indexes, so
search the web for those directly.

## Honest limits

The scanner emits **mechanistic flags, not calibrated probabilities**. Its thresholds are
judgement calls from mechanism, not fitted values. The evidence base in the catalogue is
weighted toward short-to-medium synthetic peptides; therapeutic peptides are routinely 30–40
residues and much of this transfers only partly. Say which axis a conclusion rests on and what
scope it was established at, rather than implying precision you do not have.
