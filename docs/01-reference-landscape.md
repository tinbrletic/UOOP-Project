# Reference landscape — tools and prior art

Catalogue of the external resources gathered for this project, with an honest note on
what each one actually gives us.

Verification key: **[verified]** = fetched and read during this review.
**[unverified]** = listed from the provided link, contents not independently confirmed.

---

## 1. Peptide Companion — Michael Lebl, Spyder Institute Praha

- Link: <https://www.spyderinstitute.com/software.html>
- Form: Excel workbook / desktop utility. **[unverified]**

Bench-chemist calculator for planning an SPPS run: reagent quantities, scale, molecular
weight and composition bookkeeping for a given sequence.

**What it contributes here:** it is a *calculator*, not a predictor. It does not score a
sequence for difficulty. Useful as a source of the quantities a synthesis planner cares
about, and as evidence of which sequence properties practitioners compute by hand — but it
supplies no labels, no model, and no feature set we can borrow directly.

**Do not treat it as a modelling baseline.**

---

## 2. PepFuNN — Novo Nordisk Research **[verified]**

- Link: <https://github.com/novonordisk-research/pepfunn>
- License: MIT. Dependencies: RDKit, BioPython.
- Install: `pip install git+https://github.com/novonordisk-research/pepfunn.git`
- Source under `src/pepfunn`; per-module notebooks in `notebooks/`; tests in `tests/`.

Python package for analysing natural and modified peptides: library design, clustering,
similarity analysis, and handling of non-natural monomers (BILN/HELM-style representations).

**What it contributes here:** a maintained, permissively licensed implementation for peptide
representation and similarity. Its documented modules are **library design, clustering and
similarity** — the README does not advertise a synthesis-feasibility or liability module.
Any claim that PepFuNN "scores synthesisability" must be checked against the source before
being repeated.

Its real value to us is **sequence-similarity tooling**: useful for checking whether our
train/test splits are leaking near-duplicate sequences (see audit doc, Finding 7).

Cite: PepFuNN (2025, *J. Pept. Sci.*), PepFun 2.0 (2023), PepFun (2021).

---

## 3. DLOmix / dlomix-resources — Wilhelm Lab, TU München

- Link: <https://github.com/wilhelm-lab/dlomix-resources>
- Form: Keras/TensorFlow framework + tutorial notebooks for deep learning in proteomics. **[unverified]**

Models for **retention time** and **fragment-ion intensity** prediction from peptide sequence.

**What it contributes here:** not a synthesis model — it is a mass-spectrometry model. But it
is directly relevant for a reason that matters a great deal to this project:

> Our ground-truth label is derived from MS1 peak abundance (see
> [02-source-dataset-and-label.md](02-source-dataset-and-label.md)). MS response is strongly
> sequence-dependent. DLOmix-class models exist precisely because sequence predicts MS
> behaviour. That is the mechanism behind the **ionisation confound** described in the audit.

So DLOmix is best used here as a *confound diagnostic*, not as an architecture to copy.

---

## 4. Peptimizer — MIT Learning Matter (Coley/Gómez-Bombarelli orbit) **[verified]**

- Link: <https://github.com/learningmatter-mit/peptimizer>
- License: MIT. Python 3.7, TensorFlow 2.0, RDKit.
- Notebooks: `Tutorial_CPP.ipynb`, `Tutorial_Synthesis.ipynb`.

Two frameworks:
1. **Functionality optimisation** — RNN generator + CNN predictor + genetic algorithm, for
   cell-penetrating polymers. Sequences encoded as monomer-fingerprint matrices.
2. **Synthetic accessibility optimisation** — predictor + brute-force single-point-mutation
   optimiser for **flow-chemistry** peptide synthesis.

**What it contributes here — and the most important structural lesson in this document:**

Peptimizer's synthesis predictor takes **the pre-synthesised chain, the incoming monomer,
temperature, flow rate and catalyst**. That is a *per-coupling, positional, process-aware*
model. It predicts how one step goes, given the chain built so far.

Our project instead computes **whole-sequence aggregate descriptors** (mean hydrophobicity,
total counts) and predicts one global outcome. Aggregates destroy position information.
A peptide with five scattered hydrophobic residues and one with five consecutive hydrophobic
residues get **identical** values for every mean-based and count-based descriptor we compute —
yet they behave completely differently on resin.

This is the single biggest representational gap in the current feature set, and the source
paper's own benchmark confirms it matters (it includes an `h stretch` run-length feature that
our pipeline never implemented). See audit Finding 6.

---

## 5. peptide.bio

- Link: <https://peptide.bio/> **[unverified]**

Commercial/web peptide design and analysis service. Listed for completeness. Treat marketing
claims about "synthesisability scores" as unvalidated unless a method and a benchmark are
published — no peer-reviewed benchmark has been confirmed for it in this review.

---

## Summary: what prior art actually tells us

| Question | Answer from the landscape |
|---|---|
| Is there an off-the-shelf sequence→synthesisability model we can just use? | **No.** Peptimizer is process-aware and flow-specific; PepFuNN does representation, not prediction. |
| What representation does the strongest prior work use? | **Positional / per-coupling**, not whole-sequence aggregates. |
| What is the realistic performance ceiling? | **AUC ≈ 0.77** on our dataset (source paper's own best). |
| What is the main hidden risk in our label? | **MS-derived label is sequence-biased** — DLOmix exists because sequence predicts MS response. |
