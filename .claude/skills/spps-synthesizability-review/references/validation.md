# How this skill was tested

Three rounds of adversarial testing, each on a source the skill had not been built from, each
of which found real defects that were then fixed. This file records what was done, what was
found, and — importantly — where the tests were weak.

Nothing here is a claim that the skill is correct. It is a record of what it has survived.

---

## Round 1 — A multi-author reference work

**Source.** Houben-Weyl, *Methods in Organic Chemistry* Vol. E22a–c, *Synthesis of Peptides
and Peptidomimetics*, Thieme 1999. ~2,500 pages across three scanned volumes.

**Method.** A fresh agent with no prior context was given the skill and the volumes, asked to
review them, and then asked to criticise the skill. It was told nothing about what to find.

**Result: 16 defects, all fixed.** The four that mattered:

| Defect | Fix |
|---|---|
| "Textbook" treated as one source. Applied literally, "cite for mechanism, not effect size" would have **discarded the two grade-A evidence bodies in the work** — controlled racemization kinetics and eight years of interlaboratory round-robins | Grade per **body of results**, not per document |
| Step 2 was an unsatisfiable hard gate ("if you cannot find the outcome definition, stop") on a work with no single definition | Step 2 is now per claim-cluster |
| The hydrophobic-run rule was {I,L,M,V}, so **decaalanine — the canonical difficult sequence, with NMR and resin-shrinkage evidence — scored zero** | Added `agg.homopolymer_run`, `agg.no_disruptor_window`, `agg.sheet_propensity_window`, encoding backbone regularity rather than widening the hydrophobic set |
| The racemization list named His/Cys/N-alkylated. The only controlled kinetic dataset in the source puts **Ser highest**, then Cys and Asp — Ser and Asp were absent entirely | Replaced with the measured ordering plus the mechanism that generates it |

The old list had been assembled from *which residues get special protecting groups in
practice*, not from measured rates — a failure mode worth naming because it is invisible from
inside.

---

## Round 2 — A codebase, with a no-skill control

**Source.** `learningmatter-mit/peptimizer` (MIT), a public ML repository containing a
flow-SPPS synthesis model and 12,600 instrumented coupling steps.

**Method.** Two fresh agents, identical prompts except one line: one was given the skill, one
was told to work from its own judgement. Neither knew the other existed.

**Result.** The skill arm won, and won on content traceable to specific skill steps. It made
findings the control did not:

- **The decisive one.** `prev_area` (per-step deprotection peak ratio) is flat at ~0.99 at
  every cycle from 2 to 43, while `first_area` decays 0.945 → 0.628. So the repo's
  "Pre/Post Aggregation" curve is **pure multiplicative yield loss** — no onset, no window,
  no collapse. The control found the trend; only the skill arm decomposed it and caught the
  conflation. That distinction had been added to the catalogue in Round 1.
- **7 wrong monomer structures vs the control's 3.** The extra four (D/E/S/T/Y with the tBu on
  the α-carboxyl rather than the side chain) are formula-identical with inverted connectivity,
  so the control's heavy-atom counting passed them.
- Calibrated the rules, then **stratified**, showing length AUC 0.754 → 0.578 within a band
  while windowed hydrophobicity held at 0.640.
- Explained the aspartimide null as assay blindness rather than absent chemistry.

**The control won one:** the repo's two halves do not join up — the activity optimiser has no
synthesizability term and the alphabets are incompatible. The skill arm missed it.

**Where this test was weak.** The prompt told *both* arms to note what would make them
cautious, to mark their own background knowledge, and to cite file and line. Those three
instructions encode three of the six scoring criteria, so the control was not unaided and its
showing overstates what a naive prompt produces. The contamination favoured the control, so
the skill's margin is if anything understated.

---

## Round 3 — A primary paper, full skill vs trimmed skill

**Source.** Tamás B, Alberts M, Laino T, Hartrampf N, "Amino acid composition drives
aggregation during peptide synthesis", *Nature Chemistry* 18(4):677–685, 2026,
doi:10.1038/s41557-026-02090-0. Retrieved from PubMed Central.

Chosen because it is a primary research paper (the content type the first two rounds did not
cover) and because **it contradicts the skill's own central methodological claim** with
experiments: the skill argues position beats composition; the paper reports composition
dominating, backed by 40 synthesised peptides.

**Method.** Two fresh agents, identical prompts, differing only in which method file they
read: the full SKILL.md (3,286 words) or a trimmed variant (1,026 words) with nine advisory
sections removed. References identical.

**Result: the trimmed arm matched and arguably beat the full arm.**

Both independently found: the five non-poolable outcome definitions; that the headline
crude-purity numbers have no stated measurement method anywhere; the His/Cys label artefact;
the unstated 50.1% majority baseline; accuracy-only reporting; two-laboratory pooling with
differing base rates and no stratification; the protecting-group insight; the shuffle flip
asymmetry; absent replicates; both leakage routes.

Unique to the full arm: it ran `--selftest` to confirm the rule set was behaving *before*
treating paper-vs-catalogue disagreements as real conflicts; the "label is instrument-derived"
route for laboratory effects; and that shuffling has zero power on low-complexity sequences.

Unique to the trimmed arm, and sharper: **reagent equivalents are a deterministic function of
residue identity in that dataset** (6.4 ml for one residue set, 10.4 ml for another), so no
per-residue attribution is separable from a per-residue protocol effect. Also: the paper
dismisses a competitor's metric using null-predictor reasoning it never applies to its own
58–60% against a 50.1% baseline; and it frames that competitor as limited at 60% while its own
best is 59.5%.

**What changed as a result.** SKILL.md was cut from 3,286 to ~1,950 words. The per-source
playbooks, claims-to-challenge table and tools section were removed; the evidence ladder and
the weak-result table were kept. The catalogue gained §1.0 (the protected residue is the unit),
the composition-vs-order open question, the aromatics caveat, and the pseudoproline
liability/opportunity correction.

**Where this test was weak — three ways, all disclosed:**

1. **The trimmed file was broken.** It asked for evidence grades while omitting the ladder that
   defines them. The trimmed arm noticed, read the ladder from the full file, and said so. It
   therefore had partial access to full-skill content. The ladder is retained in the trimmed
   version now shipped.
2. **The word cut overstates the content cut.** An earlier refactor had already moved the audit
   checklist and OCR protocol into `review-checklists.md`, which *both* arms had. So the test
   measured the value of SKILL.md's advisory prose, not of that material.
3. **Reference discovery was not tested.** Both arms were handed the reference file paths
   directly. Whether an agent finds them through SKILL.md alone is untested.

---

## What the testing does not establish

- **n = 1 per arm in Rounds 2 and 3.** Agent-to-agent variance could account for the
  differences. The trimmed version is *not shown to be better*; it is shown not to be worse in
  one paired comparison, which is why the cut was made but the claim is kept modest.
- **No wording micro-tests.** Whether specific phrasings bind under competing incentives has
  not been measured.
- **The rules are mechanistic, not fitted.** Thresholds come from published mechanism, and two
  (`agg.anchor_zone`, `agg.beta_branched_run`) scored near chance on the one labelled dataset
  available — for reasons the calibration output now names: too few firing rows, or a feature
  definition with no room to vary on short peptides. Untested, not refuted.

## Self-check the skill runs on itself

Calibration measures its own severity tiers rather than assuming them. On a labelled set of
1771 peptides at a 13.1% base failure rate:

| worst tier | n | fail rate | vs base |
|---|---|---|---|
| CRITICAL | 73 | 0.205 | 1.57× |
| HIGH | 843 | 0.163 | 1.24× |
| MEDIUM | 682 | 0.106 | **0.81× — at or below base** |
| INFO | 139 | 0.029 | 0.22× |

That MEDIUM sits below the base rate is why the default listing floor is HIGH, and why the
output says so instead of quietly flagging 90% of sequences. Run
`--label-column` on your own data before trusting any tier.
