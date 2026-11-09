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

Three consequences drive everything:

- **Yield is multiplicative.** At 99% per cycle a 30-mer gives ~74% crude; at 95% it gives
  ~21%. Small per-cycle losses compound into failure.
- **Failures are mostly not "the bond won't form."** The dominant cause is **loss of
  solvation**: sequence-dependent intermolecular association into β-sheet-like structures
  buries the reactive termini so reagents cannot reach them. This is the classical "difficult
  sequence" phenomenon. Everything else is a discrete, nameable side reaction with a known
  mitigation.
- **The molecule present during synthesis is not the molecule you ordered.** Twelve of the
  twenty residues carry side-chain protecting groups, the C-terminus is anchored to a polymer,
  and the solvent is aprotic. So **the protected residue, not the residue, is the unit of
  behaviour** — Ser(tBu) and Thr(tBu) behave aliphatically because the tBu ether mimics an
  aliphatic side chain, and aromatics that promote aggregation in water can reduce it on
  resin. Never reason from the free amino acid.

**A correction to the field's own vocabulary.** The language is coupling-centric — "coupling
efficiency", "difficult *coupling*", per-coupling models. But modern reagents make acylation
failure largely confined to hindered amines, and the reference literature attributes deletion
sequences in Fmoc SPPS mainly to **incomplete Fmoc removal**. One large interlaboratory study
found post-assembly steps more significant than assembly itself. When a source says
"coupling", check which step it measured.

## The four axes — what you are looking for

| Axis | What it covers | Portable across labs? |
|---|---|---|
| **Sequence-intrinsic** | Computable from the sequence (plus its protecting-group scheme) alone: hydrophobic and homopolymer runs, β-branched residues, motifs (Asp-Xaa, C-terminal Pro, N-terminal Gln), position relative to each terminus, charge | **Yes.** Transfers between labs and instruments. |
| **Process / environment** | Resin, loading and linker, coupling reagent and base, solvent, temperature, excess and cycle time, deprotection regime, capping, cleavage cocktail, flow | **Partly.** Depends on equipment and protocol. |
| **Analytical / measurement** | How "success" was defined and measured: crude HPLC purity, MS peak abundance, UV deprotection-peak shape, isolated yield, resin colour test, and any threshold applied | **No** — but it determines what any dataset can and cannot show. |
| **Provenance / operator** | How many laboratories, instruments or operators contributed; central or per-site characterisation; whether provenance is recorded at all | **No.** Often the largest effect, and usually unrecorded. |

**A claim is only as transferable as its weakest axis.** A sequence rule validated on one
instrument under one coupling protocol is a claim about that protocol too.

The fourth axis exists because the strongest interlaboratory evidence concludes quality
depends as much on personnel as on method, and in one round-robin all the high racemization
values came from a single laboratory. Note it has a second route that is easy to miss: when
the **label itself is instrument-derived** — a threshold on a UV trace, say — a difference in
detector baseline or flow behaviour between two builds shifts the label directly, with no
operator involved.

| Reference | Use it for |
|---|---|
| `references/factor-catalogue.md` | Every known factor on all four axes, with mechanism, the vocabulary sources use for it, and whether it is detectable before synthesis. **Read before reviewing unfamiliar material.** |
| `references/review-checklists.md` | The dataset/model audit (nine ordered steps) and the protocol for reading scanned sources without misquoting them. |
| `references/worked-example.md` | A worked end-to-end review. |
| `references/validation.md` | How this skill was tested, what the tests found, and where they were weak. Read before relying on any rule or effect size. |

## Workflow

**1. Scope.** Write down the question before reading.

**2. Locate the outcome definition for each distinct body of results — before reading for
conclusions.** This step does more work than all the others together.

A source usually has several, and they are **not poolable**: a resin colour-test "complete
coupling", a crude HPLC area fraction at 210 nm, a UV deprotection-peak shape and a
chiral-HPLC half-life are four different quantities. Tabulate them. Sometimes a definition
must be assembled — the detection wavelength that defines a purity measure can sit in a
figure caption while the protocol sits in the text.

**When a body of results has no locatable definition, record that and treat its numbers as
incomparable to everything else** — do not discard the source, and do not quietly assume the
definition. Watch especially for a single asserted sentence bridging a measured proxy to the
quantity a practitioner cares about; that bridge is often the least supported sentence in the
work and the one everything else rests on.

**3. Read for factors, not for conclusions.** Record each asserted factor as a finding record
(below).

**4. Grade the evidence — per body of results, not per document.** The unit is a coherent set
of measurements with one outcome definition, which in a primary paper is usually a results
section and in a multi-author volume is a chapter. The same work can honestly carry a B and a
D at once. Grading a document as a unit either throws away its best evidence or launders its
worst.

**5. Cross-check — including against the source itself.** For each factor ask: does another
source contradict it; **does this source contradict itself**; and can the claimed mechanism
and the measurement actually see each other?

The self-contradiction check is the highest-yield step in the workflow, and the richest seam
is **reading Methods against Results**. A sample-handling decision that looks like
housekeeping in Methods can be a predicted bias in a headline result three sections earlier.
In multi-author works expect the mechanism chapters and the data chapters to disagree, and
prefer the measurement.

**6. Synthesize** into the four axes, separating what is established from what one source
asserts once.

## Finding record

```
factor        what is claimed to affect outcome
axis          sequence-intrinsic | process-environment | analytical-measurement | provenance
direction     helps | harms | non-monotone | unclear  -- PER AXIS, not one verdict
tradeoff      the countervailing finding, if this factor helps on one axis and harms on
              another; else "none known"
mechanism     why, in one sentence - or "asserted, no mechanism given"
evidence      grade for THIS BODY OF RESULTS, not the whole work
scope         sequence lengths, chemistry, phase, instrument, protecting-group scheme and
              era the claim was established on
detectable    computable from sequence alone? needs process metadata? measured post hoc?
conflicts     other sources OR OTHER SECTIONS OF THIS SOURCE that disagree
reconstruction  "verbatim", or "reassembled from OCR fragments - verified by <method>"
```

**`direction` is per axis, and `tradeoff` exists because one verdict is usually wrong.** The
normal shape of an SPPS decision is a residue that helps one thing and harms another at the
same position: Pro, Arg, His and Gln are reported as protective against assembly failure
while Arg (δ-lactam), His (epimerisation) and Gln (pyroglutamate, dehydration) are *also*
named liabilities. A residue can even be the top liability **and** the repair site — Ser and
Thr drive aggregation and are the only positions where a pseudoproline can be installed.

**`detectable` matters more than it looks.** Features derived from observed spectra — which
residue was lost in a failed run — cannot score a peptide that has not been made yet. Sources
mix these in with predictive features freely.

## Evidence ladder

| Grade | What it takes |
|---|---|
| **A** | Controlled experiment isolating the factor, replicated, with the outcome measure stated |
| **B** | Large labelled dataset with held-out validation **and a stated baseline** |
| **C** | Retrospective association in one dataset, no baseline or no held-out set |
| **D** | Mechanistic argument or worked examples, no systematic data — most textbook and vendor content |
| **E** | Assertion by citation chain, where the cited source does not support the specific claim |

Grade D is not worthless; most reliable SPPS mechanism sits there. The failure is citing D as
if it were A. **The baseline requirement in B is load-bearing**: a source reporting accuracy
without its majority-class rate has not shown you an effect size, and model-interpretation
outputs (feature importances, SHAP values) inherit the credibility of the model they explain.

## Screening sequences

When the content contains sequences, run the scanner rather than reading them by eye. Check
first that it does — a source naming peptides by protein fragment carries no sequences at all,
and declining to run a tool is a reportable decision.

```bash
python scripts/scan_peptides.py <file>                        # CSV/TSV/XLSX/FASTA/text
python scripts/scan_peptides.py <file> --label-column <col>   # calibrate on your own data
python scripts/scan_peptides.py --selftest                    # known-difficult + controls
```

Output splits into **chain-assembly difficulty** and **process liabilities**, which answer
different questions. The scanner ships **no hardcoded effect sizes** — predictive power is a
property of a dataset and its outcome definition, not of the chemistry. `--help` for options.

### Reading a weak result: untested, degenerate, invisible — rarely refuted

A lift near 1.0 is the least informative number in the output, because four different things
produce it. Calibration flags the first two; you reason about the others.

| Signal | What it means | Flagged as |
|---|---|---|
| **Untested** | Too few firing rows for any effect to appear. A rule firing on 3 of 1771 rows at a 13% base rate expects 0.4 failures — its lift is counting noise. | `UNTESTED (exp. N failures)` |
| **Degenerate** | The *feature definition* has no room to vary here. An anchored window 8–25 residues from a terminus is a few residues wide on a 15-mer, empty on a 9-mer. | `DEGENERATE (N% zero)` |
| **Invisible** | Real and common, but the outcome measure cannot see it — an MS1 purity label and an isobaric by-product. | nothing; reason from step 2 |
| **Refuted** | The mechanism does not operate in this sequence space. | only after excluding the above |

**Do not retune a mechanistic threshold so it fires on the dataset in front of you.** A
threshold set from chemistry is a claim, not a hyperparameter. Report the rule as untested on
that data and leave it alone.

**Distinguish a rule from the feature underneath it.** A longest-run *feature* can carry real
signal across its range while the *rule* firing at run ≥ 5 is untestable because almost
nothing reaches 5.

**When there are no labelled outcomes**, run `--selftest` to confirm the rule set is behaving
before treating any disagreement with a source as a real conflict rather than a rule defect.
Then run the scanner on the failure cases the source itself names — free validity checking
against ground truth the author hands you.

## Honest limits

The scanner emits **mechanistic flags, not calibrated probabilities**; its thresholds are
judgement calls from mechanism, not fitted values. The catalogue's evidence base is weighted
toward short-to-medium synthetic peptides, and therapeutic peptides run 30–40 residues. Say
which axis a conclusion rests on and what scope it was established at, rather than implying
precision you do not have.
