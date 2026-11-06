# Houben-Weyl E22a–c — review of factors affecting peptide synthesis outcome

> ## Provenance — read before citing anything below
>
> **This document is machine-generated output that has been only partially verified.**
> It is kept because its factor inventory and its catch of six internal contradictions in
> the source are genuinely useful, not because it is authoritative.
>
> **How it was produced.** A Claude Code agent with no prior context was given the
> `spps-synthesizability-review` skill and the three reference volumes in
> `docs/Marta de Zotti - materials/`, and asked to review them and then criticise the skill.
> It extracted ~2497 pages with `pdftotext`, built a page index, and read located sections.
> It was read-only and modified nothing in the repository.
>
> ### Independently verified (3 claims)
>
> These were re-checked against the extracted source text by a second pass before this file
> was committed, and are confirmed:
>
> | Claim | Status |
> |---|---|
> | Decaalanine failed on polystyrene; critical region "the zone about 8-10 residues out from the resin"; β-structure, severe resin shrinkage, NMR line-broadening (E22a) | **Verbatim confirmed** |
> | "The synthesis will rarely collapse after 25 residues" (E22b) | **Verbatim confirmed** |
> | Racemization propensity — high Ser/Cys/Asp, moderate Phe/Ala, low Val/Ile — with the β-electron-withdrawing / β-branching mechanism stated beneath the table; measured by chiral HPLC half-lives (E22b) | **Confirmed, including the OCR header scramble the report flags** |
>
> ### NOT verified
>
> **Everything else**, including: all other page citations; the interlaboratory percentages
> in the Axis-4 / operator section; the Fmoc-removal half-life table; the resin-swelling
> figures; the "0.3% per deprotection cycle" aspartimide figure (which the report itself
> flags as a single unreplicated citation); and the attribution of specific claims to
> specific chapter authors. Check against the PDFs before citing any of it in a report or
> paper.
>
> ### Two caveats the report raises about itself
>
> 1. **Quotations are reconstructions.** The OCR reflows two-column pages into near-random
>    line order and joins words. Every quotation was reassembled from adjacent fragments on
>    one page and verified by fixed-string search, but that is weaker than transcription.
>    Where reconstruction was uncertain the report paraphrases instead — but the distinction
>    is carried in prose, not marked per sentence.
> 2. **Page numbers are PDF page numbers** in the files as supplied, not the printed page
>    numbers of the published volumes. The two differ. A reader cannot check a citation
>    without knowing which is meant.
>
> ### Status of Part 2
>
> Part 2 is the agent's criticism of the skill, not a finding about peptide chemistry. All
> sixteen defects it raises **have been fixed** in
> `.claude/skills/spps-synthesizability-review/`. It is retained as the record of why the
> skill looks the way it does. One of its recommendations was deliberately **not** followed
> as written: it proposed adding Ala to a β-sheet-propensity set, but Ala's Chou-Fasman
> P<sub>β</sub> is low (≈0.83), so the fix encodes backbone *regularity* instead. See
> [tinbrletic/UOOP-Project#2](https://github.com/tinbrletic/UOOP-Project/pull/2).
>
> ### Source and copyright
>
> *Houben-Weyl Methods in Organic Chemistry*, Vol. E22, **Synthesis of Peptides and
> Peptidomimetics**, Thieme, 1999 — a copyrighted reference work. Quotation here is short,
> attributed, and for the purpose of criticism and review. Do not redistribute this document
> as a substitute for the source.

---

## Part 1 — Review: factors affecting peptide synthesis outcome

Produced by following the `spps-synthesizability-review` skill. Read-only throughout; nothing was written to the repository. Intermediate working files (extracted text, a page index, an extracted sequence FASTA) were kept in a session scratchpad and are not preserved — regenerate them with `pdftotext` on the volumes in `docs/Marta de Zotti - materials/` if you need to re-check a citation.

**Method.** `pdftotext` on all three volumes (2497 pages), reindexed to one `vol<TAB>page<TAB>line` file, searched on single tokens as advised. Built a heading index from the numbered section lines to locate chapters, then read the hot-spot sections in full. Page numbers below are **PDF page numbers** in the supplied files, with the Houben-Weyl section number alongside.

**OCR caveat that affects quoting.** The text layer reflows two-column pages into near-random line order and joins words. Every quotation below is a reconstruction from adjacent fragments on one page; I verified each by `grep -F` on the distinctive substring. Where reconstruction was uncertain I paraphrased instead of quoting. One OCR failure is material: the column headers of Vol B Table 4 are emitted as "Moderate / Low / High" while the data row reads "Ser,Cys,Asp | Phe,Ala | Val,Ile". I assigned High→Ser,Cys,Asp on two independent grounds — the stated mechanism (β-electron-withdrawing increases, β-branching decreases) and the half-lives in Table 1 of the same section (Ser(OBzl) 1.7 min < Asp(OBzl) 6.0 min < Phe 16 min << Val 150 h). I would not have trusted the table alone.

---

## Source

> *Houben-Weyl Methods in Organic Chemistry*, Vol. E22, **Synthesis of Peptides and Peptidomimetics**, Thieme, 1999. Volumes E22a, E22b, E22c (E22d/e not supplied).
> **Type:** `textbook` / multi-author reference work — but see below, this classification is the single biggest problem the skill handed me.
> **Outcome measure:** varies by chapter. No work-level definition exists.
> **Threshold:** varies; 75%, 70% and 50% "correct product" all appear.

### Step 2 — outcome definitions, located per chapter

The skill says find the outcome definition before anything else and stop treating the numbers as comparable if you cannot. In a 2497-page multi-author reference work there is no single definition, so I located one per chapter that reports numbers. This is the step that most changed the review:

| Chapter | Outcome measure | Threshold |
|---|---|---|
| **B 7.8** (Bonewald) PSRG/ABRF interlaboratory studies | "% correct product" by RP-HPLC + MS, centrally characterised, one target peptide per year across many labs | 75%, 70%, 50% all used |
| **A 4.3.4** (Rivier & Miranda) elevated temperature | crude peptide **weight** recovery + RP-HPLC purity, UV **210 nm**; purified identity by MALDI-MS | none; visual HPLC comparison |
| **A 4.3.2** (Stewart) Boc/Bzl protocols | **Kaiser (ninhydrin) colour test** on resin beads, read by eye against white background | semi-quantitative: faint blue ≈ 99% complete, dark blue ≈ ≤75% complete (A p737) |
| **A 4.3.2.2.5** Fmoc monitoring | continuous-flow UV of dibenzofulvene release; **peak shape** read qualitatively | none — explicitly could not reach the accuracy needed for 100%/step (A p747) |
| **B 7.4** (Romoff) racemization | **L/D ratio by chiral HPLC** after benzylamine quench → half-life *t*<sub>R½</sub> | continuous, no threshold |
| **A 2.2.2**, **A 4.3.2.2.7**, **B 5.3.1** mechanism chapters | none — no outcome data at all | — |

Three consequences I recorded before reading further:

1. **Nothing in this work is cross-chapter comparable.** A Kaiser-test "complete coupling", a 210 nm crude HPLC area fraction and a chiral-HPLC half-life are three different quantities. I did not pool them anywhere.
2. **The Kaiser test is the outcome measure for most of the Boc protocol chapter, and it is self-confessedly blind in exactly the cases those protocols are for** (see Finding 3). That is the most load-bearing analytical fact in the whole source.
3. **"Synthesis failure" here genuinely does sometimes mean "cannot be made"**, not merely low crude purity — DKP cleaves the chain off the resin, pyroglutamate and guanidination-type events cap it irreversibly. This is the one place where the skill's standing correction ("almost always means low crude purity") is itself wrong for the source, and I say so under Assessment.

---

## Factors claimed

Axis: **S** = sequence-intrinsic, **P** = process/environment, **A** = analytical/measurement. Evidence grades per the skill's ladder.

| # | Factor | Axis | Direction | Evidence | Where |
|---|---|---|---|---|---|
| 1 | On-resin aggregation from hydrophobic runs → β-sheet, resin shrinkage, incomplete acylation **and** incomplete Fmoc removal | S | harms | **D** | A p726 (4.3.2); A p749–750 (4.3.2.2.7.3); B p64 (5.3.1) |
| 2 | **Onset of collapse at 8–12 residues from the resin; "rarely collapses after 25 residues"** | S/P | **non-monotone** | **D** | B p66 (5.3.1); A p726 gives "8–10 residues out from the resin" |
| 3 | Kaiser test gives **false negatives** under aggregation or steric hindrance | **A** | — | **D** | A p737 (4.3.2.1.3.1) |
| 4 | Racemization rate vs **side chain**: high Ser, Cys, Asp; moderate Phe, Ala; low Val, Ile | S | harms | **A** | B p665–667 (7.4), Tables 1 & 4 |
| 5 | Racemization rate vs **base**: high TEA; moderate DIPEA; low 4-alkylmorpholines, 2,4,6-collidine | P | harms | **A** | B p666–667 (7.4), Tables 2 & 4 |
| 6 | Racemization rate vs **solvent**: high DMF **and CH₂Cl₂**; low THF, toluene | P | harms | **A** | B p666–667 (7.4), Tables 3 & 4 |
| 7 | Epimerization is suppressed by **increasing coupling rate** (competition kinetics) | P | helps | **D** | B p660, p667 (7.4) |
| 8 | Aspartimide at Asp-Xaa, **up to 0.3% per deprotection cycle for Asp-Gly** | S | harms | **D** | A p749 (4.3.2.2.7.1) |
| 9 | Aspartimide occurs by **both** acid (HF cleavage, Boc/Bzl) and base (piperidine, Fmoc) routes | S/P | harms | **D** | A p238 (2.2.2) |
| 10 | Aspartimide/β-aspartyl by-products are **chromatographically hard to separate** from target | **A** | — | **D** | A p238 |
| 11 | DKP (piperazine-2,5-dione) chain loss at C-terminal dipeptide; worst with Pro/N-alkyl | S | harms | **D** | A p728; A p749 (4.3.2.2.7.2) |
| 12 | DKP **does not occur** with Rink-type amide or trityl-based ester linkers | P | helps | **D** | A p749 |
| 13 | Pyroglutamate from N-terminal Gln caps the chain; **observed** in YLRPDWHQA at elevated temperature | S/P | harms | **C** | A p729; A p811 (4.3.4.4) |
| 14 | **Pro is the most effective residue for preventing synthesis failure**; then Arg(Pmc), His(Trt), then Gln(Trt), Asn | S | **helps** | **D** | B p66 (5.3.1) |
| 15 | Backbone amide protection (Hmb, Dmb) at **positions 6, 12, 18, 24** — every ≤6 residues | P | helps | **D** | B p66; A p727, p750 |
| 16 | Hmb **costs** a coupling: β-branched residues onto Hmb need >24 h + repeat couplings, except Hmb-Gly | P | harms | **D** | B p64, p66 |
| 17 | Pseudoprolines (oxazolidine/thiazolidine from **Ser, Thr, Cys**) increase chain solvation, disrupt β-sheet | S | helps | **D** | A p750, p802 |
| 18 | Elevated temperature: **60 °C near-optimal, 75 °C not reliably so**; effect on aggregation explicitly uncertain | P | **unclear** | **C** | A p806–810 (4.3.4) |
| 19 | Resin swelling by solvent, measured in triplicate at 27/60 °C; 25% DMSO/toluene and NMP maximal, DMF lower and *falling* with temperature | P | helps | **B** | A p808 (4.3.4.1), Table 1 |
| 20 | Low resin loading (<0.15 mequiv/g; 0.04–0.3 mmol/g for convergent; ~50 µmol/g for segment condensation) | P | helps | **D** | A p806, p790; B p66 |
| 21 | Chaotropic salts (NaClO₄ > KSCN) break up secondary structure | P | helps | **D**, contested | A p727, p736 vs A p746 |
| 22 | Aged/contaminated DMF (dimethylamine) → premature deprotection; DNFB test given | P | harms | **D** | A p738 (4.3.2.1.6) |
| 23 | Excess base + TBTU/DIPEA → premature Fmoc removal → **multiple addition** (double insertion) | P | harms | **D** | A p750 (4.3.2.2.7.4) |
| 24 | C-terminal Cys ester: β-elimination → dehydroalanine → piperidine adduct; **Trt protection virtually avoids it, Acm does not** | S/P | harms | **D** | A p748 (4.3.2.2.7.1) |
| 25 | C-terminal Cys ester racemizes under **repetitive** Fmoc removal (cumulative in cycle count) | S/P | harms | **D** | A p751 |
| 26 | Prolonged activation → **dehydration of Gln and Asn**; recouple rather than extend coupling | P | harms | **D** | A p811 |
| 27 | Capping (Ac₂O / acetylimidazole) converts deletions to truncations | P | helps | **D** | A p729, p738, p811 |
| 28 | Segment condensation: C-terminal residue of the activated segment **classically limited to Gly or Pro** | S/P | harms | **D** | B p66; A p803 |
| 29 | Segment-condensation deletions differ by a whole segment → easy to purify, unlike single-residue deletions | **A** | helps | **D** | B p65 |
| 30 | **Coupling time showed no correlation with % correct product** across 33 labs | P | **none found** | **C** | B p771 (7.8.4.3) |
| 31 | **Post-assembly side reactions (cleavage + deprotection) were more significant than assembly itself** | P | harms | **C** | B p766 (7.8) |
| 32 | **Operator experience and skill matter as much as or more than instrument or method** | *(no axis)* | — | **C** | B p766 |
| 33 | MS + HPLC alone **cannot detect a reversed sequence of equal mass**; 7/20 labs passed a scrambled peptide as correct | **A** | — | **A** | B p771 (7.8.4.4) |
| 34 | Deamidation is **position-dependent**: same Gln deamidated at position 3 from the C-terminus, not at position 3 from the N-terminus | S | harms | **A** | B p772 |
| 35 | Deamidation **not detectable** by MALDI-TOF or quadrupole ESI; needs ion-trap ESI | **A** | — | **C** | B p772 |
| 36 | AAA/Edman chemistry **cleaves residual protecting groups, making a poor synthesis appear correct** | **A** | — | **D** | B p768 |

---

## Finding records — the load-bearing eight

```
factor      Positional window of aggregation onset: 8-12 residues from the resin,
            "rarely collapses after 25 residues"
axis        sequence-intrinsic (but measured from the C-terminal anchor, so it is
            a resin-relative position, not a position in the free peptide)
direction   NON-MONOTONE in chain length - rises into a window, then falls
mechanism   beta-sheet association needs enough chain to nucleate but, past ~25
            residues, the already-collapsed matrix is not what is being titrated;
            the authors assert the observation and not the mechanism
evidence    D - stated as an in-house planning rule validated by use ("when these
            rules were adhered to no redesigning has been required"), with no n,
            no controls, no counts. Two independent chapters agree on the window
            (B p66 "8-12"; A p726 "zone about 8-10 residues out from the resin"),
            which is the strongest thing about it.
scope       Fmoc/tBu SPPS and solid-phase segment condensation of protein
            fragments, polystyrene and low-loading supports, 1999 reagents
detectable  YES from sequence alone, and cheaply - but only if you index position
            from the C-terminus. Whole-sequence means and whole-sequence maxima
            both destroy it.
conflicts   Contradicts the skill's own catalogue (1.1: "risk rises with length in
            principle") and the scanner's agg.length rule, which fired HIGH on a
            55-mer. HW says collapse rarely happens after 25 residues. The two
            claims are reconcilable only if "length" in the catalogue means
            cumulative-yield loss and not aggregation - which is exactly the
            conflation the catalogue warns about elsewhere.
```

```
factor      Racemization propensity by side chain: Ser, Cys, Asp high; Phe, Ala
            moderate; Val, Ile low
axis        sequence-intrinsic
direction   harms
mechanism   two opposing substituent effects at the beta position -
            electron-withdrawing groups stabilise the alpha-carbanion
            (thermodynamic, raises propensity); beta-branching sterically blocks
            the base's approach to the alpha-proton (kinetic, lowers it)
evidence    A on isolation - one variable changed at a time, outcome measure
            stated (L/D by chiral HPLC on a Chiralcel OD column, benzylamine
            quench to freeze the ratio), result reported as a half-life.
            Downgrade note: replication is not stated, and the span is huge
            (Ser(OBzl) 1.7 min vs Val 150 h, ~5000x), which is reassuring for
            ordering but means a single bad run would not be visible.
scope       Boc-Xaa-UNCA (urethane-protected N-carboxyanhydrides), SOLUTION phase,
            toluene, 20 C, TEA 1.5 equiv. Transfer to SPPS is ASSERTED in one
            sentence ("These findings also hold for solid-phase synthesis
            reactions", B p667) with no supporting data.
detectable  YES from sequence alone
conflicts   DIRECTLY contradicts Stewart in the same work: "Of all the natural
            amino acids, phenylalanine is most susceptible to racemization"
            (A p727). Romoff's measured data put Phe in the moderate band.
            Also contradicts the skill's catalogue (1.4: "most prone: His, Cys,
            and N-alkylated"), which omits Ser and Asp entirely - Ser being the
            single worst residue in the only controlled dataset in the source.
```

```
factor      The Kaiser (ninhydrin) test gives false negatives in difficult
            sequences
axis        ANALYTICAL/MEASUREMENT
direction   n/a - it is the measurement, not a factor
mechanism   "It can, however, give false negative results in 'difficult'
            sequences where there is serious aggregation or steric hindrance"
            (A p737). Also inapplicable to N-terminal secondary amines (Pro ->
            use isatin or chloranil), and N-terminal Asn, Asp, Ser give a brown
            colour that confounds the read.
evidence    D (assertion with mechanism, no quantitation of the false-negative
            rate)
scope       Boc/Bzl manual SPPS, bead colour read by eye
detectable  measured post hoc, on-resin
conflicts   none internal - and this is the point. The Kaiser test is the
            completion criterion throughout A 4.3.2's protocols ("Mix until the
            Kaiser test is negative", repeatedly), including the chaotropic-salt
            protocol for difficult sequences at A p736. So the protocol
            prescribed FOR difficult sequences is validated BY an assay the same
            chapter says is blind in difficult sequences. Any claim in that
            chapter that a procedure "goes to completion" on a difficult
            sequence is unfalsifiable as reported.
```

```
factor      Incomplete Fmoc REMOVAL, not incomplete coupling, is the main cause
            of deletion sequences in Fmoc/tBu SPPS
axis        process-environment (consequence of a sequence-intrinsic cause)
direction   harms
mechanism   aggregation collapses the peptide-resin matrix and hinders diffusion
            to and from the reactive sites; modern coupling reagents are
            efficient enough that coupling failure is "now generally limited to
            cases where the amine to be acylated is extremely hindered, such as
            N-substituted amino acids" (A p750)
evidence    D
scope       Fmoc/tBu, polystyrene especially; "less common" on polyacrylamide or
            PEG-based supports, though changing support does not fix it because
            "the root of the problem lies in the interactions between peptide
            chains"
detectable  needs process metadata; partially visible live as a broadened,
            tailing dibenzofulvene release peak in continuous-flow UV (A p747)
conflicts   Refines rather than contradicts the skill's framing. The skill says
            failures are "mostly not 'the bond won't form'" and attributes this
            to loss of solvation - correct - but then its whole vocabulary
            (coupling efficiency, difficult COUPLING, per-coupling models) is
            about the acylation step. HW says the acylation step is largely
            fixed and the DEPROTECTION step is where the losses are. That
            changes which process variables matter: piperidine regime, base
            choice, deprotection time and temperature, not reagent potency.
```

```
factor      Proline, and then Arg(Pmc)/His(Trt)/Gln(Trt)/Asn, PREVENT synthesis
            failure
axis        sequence-intrinsic
direction   HELPS - and this is the problem
mechanism   secondary amino acids and (asserted) certain bulky side-chain
            protecting groups disrupt beta-sheet formation
evidence    D (in-house planning rule, two citations, no counts)
scope       Fmoc/tBu, difficult protein-fragment sequences
detectable  YES from sequence alone, including the needed positional form
            ("in the absence of the aforementioned residues ... within the
            first six residues and then at intervals of every sixth residue
            or less")
conflicts   Internal to the review, and the skill's record format cannot hold it.
            Arg and His appear in this work BOTH as protective against assembly
            failure (B p66) AND as liabilities - Arg via delta-lactam and
            des-Arg deletions, His as the second-worst racemization risk after
            Ser/Cys/Asp and the residue needing Mbom/Bum protection (A p751).
            Gln is simultaneously protective here and the pyroglutamate risk at
            the N-terminus (A p729) and a dehydration risk on prolonged
            activation (A p811). The same residue helps on one axis and harms on
            another, and "direction: helps|harms|non-monotone|unclear" has no
            way to say that.
```

```
factor      Elevated temperature (60 C) for difficult couplings
axis        process-environment
direction   UNCLEAR, per the authors themselves
mechanism   faster reaction, and "under certain circumstances" disruption of
            deleterious secondary structure
evidence    C, and I want to be precise about why it is not higher. The chapter
            is a comparative synthesis study - ACP(65-74), YLRPDWHQA-amide and
            hCRF at 27/45/60/75 C - which looks like a controlled experiment.
            It is not: n=1 per condition; the authors state "it is difficult to
            carry out statistical analyses on synthetic procedures that
            inherently have multiple degrees of freedom" (A p810); and the 45 C
            arm, which gave "significantly poorer results", is attributed by the
            authors themselves to a protocol artefact (20-min couplings forcing
            recoupling at most residues) rather than to temperature. Solvent was
            changed alongside temperature (CH2Cl2 at 27 C vs 25% DMSO/toluene at
            60 C), so temperature and solvent are confounded by design.
scope       manual Boc and Fmoc SPPS, MBHA/PAM/TMBHA/Rink/SAMBHA resins,
            25% DMSO/toluene, <=80 C, peptides <=41 residues
detectable  needs process metadata
conflicts   Self-contradictory within one chapter, and the hedge is the honest
            reading: "The results suggest that the use of high temperature may or
            may not minimize problems associated with secondary structure
            formation that would interfere with coupling. Whether it is a case of
            appreciation or of sequence dependency will need further
            investigation" (A p807) - against, three pages later, "no
            difficulties caused by beta-sheet formation were encountered. This is
            in accordance with the hypothesis that the occurrence of aggregates
            can be avoided at high temperature" (A p810). It also notes prior
            work "used by several groups with conflicting results" (A p806).
            Separately, A p746 ranks heating to 45 C ABOVE chaotropes for
            improving Fmoc removal, while A p727/p736 present chaotropic salts as
            the remedy for difficult sequences. One work, three positions.
```

```
factor      Interlaboratory variance and operator skill
axis        NONE OF THE THREE
direction   harms
mechanism   none offered beyond the observation
evidence    C for the causal claim; the underlying distributional data are the
            best empirical material in all three volumes. ABRF Peptide Synthesis
            Research Group, 1991-1998, eight round-robin studies: one target
            peptide per year, synthesised independently by many member
            laboratories, centrally characterised by HPLC + AAA + Edman + MS.
            1991: 17% of submitted samples contained none of the correct peptide;
            28% of crude preparations and 68% of purified samples exceeded 75%
            correct peptide. 1992: 48% of crude and 21% of purified samples were
            below 75% of the desired product. 1996: of 53 peptides, two below 50%
            and three others below 70%. 1997 phosphopeptide study, 33 samples:
            20 above 75% correct, five below 50%, four with no correct product.
scope       whatever chemistry each member facility chose - which is the point
            and also the limitation
detectable  NOT detectable from sequence, NOT from process metadata as usually
            recorded. Invisible to every axis the skill defines.
conflicts   This is a cohort-structure finding of the kind the skill's audit
            section demands, arriving from the opposite direction. A pooled
            dataset assembled from facilities like these would carry a lab
            effect large enough to dominate any sequence feature, and 1996 shows
            it concretely: racemization ran 2-8% across submissions but "only
            peptides from one laboratory possessed the higher level of
            racemization" (B p770). The strongest "sequence" signal in such a
            dataset could be a lab indicator.
```

```
factor      HPLC + MS cannot detect a reversed sequence of identical mass
axis        ANALYTICAL/MEASUREMENT
direction   n/a
mechanism   mass and retention behaviour are not injective over sequence
            permutations; the PSRG deliberately built the test case by reversing
            a peptide and making two substitutions to equalise the mass
evidence    A - this one genuinely is. A controlled, blinded test with a known
            ground truth, a stated outcome (did the facility report the
            discrepancy), and a measured detection rate: of 20 participating
            laboratories, 7 assumed the reverse peptide was correctly made; the
            10 that identified the error "used Edman sequencing" (B p771). The
            same study spiked 30% Met(O) into a second peptide and only 7 of 20
            detected the oxidation. 19 of 20 used MS routinely, 16 HPLC, 7 AAA.
scope       1998, core-facility practice, short peptides (7-9 residues)
detectable  measured post hoc
conflicts   Extends the skill's Axis 3 table rather than contradicting it. The
            catalogue's MS entry warns about ionisation bias and isobaric
            by-products; this is a third and larger blind spot - the target
            itself can be the wrong molecule at the right mass. Any dataset whose
            positive label is "observed mass matches ordered peptide" contains an
            unmeasured fraction of mislabelled successes, and the 1998 study puts
            a floor under that fraction for one specific error mode.
```

---

## Process disclosed / omitted

Unusually complete, because much of the work is protocol text rather than results.

**Disclosed, with numbers:** resin type and cross-linking (1% DVB standard; swelling spec 1.0 mL/100 mg in CH₂Cl₂, A p726) · loading (<0.15, 0.04–0.3, ~0.05 mmol/g in three different contexts) · linker (PAM, Wang, Rink, trityl/CTC, HMFS, MBHA/TMBHA/SAMBHA) · coupling reagent and equivalents (DIC/DCC, BOP, PyBOP, HBTU/TBTU, HATU, TFFH, UNCAs, acid fluorides; typically 2–2.5 equiv) · base and equivalents · solvent and grade, including a DNFB assay for dimethylamine in DMF (A p738) · temperature and heating method · coupling and deprotection times (20% piperidine/DMF 1×3 min + 1×7 min) · Fmoc-removal half-lives by base (20% piperidine/DMF 7 s; 5% piperazine 20 s; 50% morpholine 60 s; 50% Cy₂NH 2100 s; 50% DIPEA 10.1 h — A p746 Table 1) · capping reagents · cleavage cocktails and scavengers · backbone-protection positions · platform (manual, continuous-flow, automated).

**Omitted or absent:**
- **Any statistical treatment.** Explicitly disclaimed at A p810. No replicates on synthesis outcomes anywhere; the only triplicates are resin-swelling volumes (A p808).
- **Per-cycle efficiency data.** The source knows quantitative Fmoc release is the right measurement and says it was never achieved to the needed accuracy (A p747). So the multiplicative-yield argument that motivates the whole field is, in this work, unmeasured.
- **Graded motif tables.** The skill's catalogue carries a nine-way graded Asp-Xaa sensitivity ranking. **Houben-Weyl has no such table.** It names Asp-Gly/Asp-Ser/Asp-His (A p238) in one chapter and Asp-Asn/Asp-Gly/Asp-Gln/Asp-Ser (B p66) in another, ungraded, with Asp-Gln in the "especially" list where the catalogue puts it near the bottom.
- **No UV-detector-response discussion.** I searched for it; Vol B's HPLC chapters (7.1, 7.2) treat retention and separation, and "molar absorptivity" appears only in a symbol glossary (B p546). The source reports crude purity at 210 nm without addressing non-uniform response. The skill's Axis 3 warning is correct and this source does not make it.
- **Sequences.** Most model peptides are named, not spelled. ACP(65-74), magainin-II-amide and hCRF are discussed repeatedly with no sequence given. I did **not** supply them from my own knowledge for the scan.

---

## Scanner run

I ran the mandated scan on the ten sequences the material actually spells out (`hw_seqs.fasta`). No labelled outcomes exist in usable form, so calibration mode was not available — flag mode only.

It behaved sensibly on most, and the ABRF 1996 racemization peptide **RERHAY** is a clean hit: the scanner flags His epimerisation, and the PSRG measured exactly this peptide and found 2–8% racemization. Right mechanism, modest effect size, concentrated in one lab.

Two failures worth recording:

**Decaalanine.** `AAAAAAAAAA` draws one MEDIUM flag from a hydrophobic-window rule, with `h_stretch 0`. This is the archetypal catastrophic failure in the source — "This caused failure of sequences as simple as decaalanine, and the critical region was the zone about 8-10 residues out from the resin" (A p726), with NMR line-broadening as the physical evidence. The scanner cannot see it because `agg.h_stretch` is defined over {I, L, M, V} and Ala is not in the set — a definition inherited straight from the catalogue's §1.1. The most famous difficult sequence in SPPS is a poly-Ala decamer and the rule misses it by construction.

**TRNLADQED.** Flagged only `side.arg_lactam` + `side.aspartimide` (LOW). This is the ABRF 1998 reverse peptide, and it is the one peptide in the source with a *measured* stability failure: its C-terminal Asp-α-amide "was highly susceptible to deamidation" and its Gln at position 3 from the C-terminus also deamidated (B p772). `stability.deamidation` is an Asn-Gly motif rule and fires on neither. The positional finding — same Gln, deamidates at 3-from-C, does not at 3-from-N — is the cleanest controlled demonstration in all three volumes of the skill's own "localisation matters more than totals" principle, and the scanner has no positional deamidation rule to express it.

---

## Assessment

**Supported — grade A, controlled, with a stated outcome measure.** Only two bodies of work in 2497 pages qualify. (i) The UNCA racemization kinetics in B 7.4: side-chain, base and solvent each isolated, half-lives measured by chiral HPLC. The ordering Ser < Asp < Phe << Val spans ~5000× and the base and solvent orderings are consistent with the stated mechanisms. (ii) The ABRF 1998 blinded analytical test in B 7.8: known ground truth, measured detection rate, 7/20 labs passed a scrambled peptide and 13/20 missed a 30% oxidised impurity.

**Supported — grade B.** The resin-swelling measurements (A p808): triplicate, quantitative, one variable at a time, at two temperatures. Note the direction nobody flags — DMF swelling *falls* from 27 °C to 60 °C for three of four resins (TMBHA 0.90→0.77, Rink 1.06→1.00, SAMBHA 0.76→0.69 mL), while DMSO/toluene rises. If poor swelling is "a cause or an indication of poor couplings" (A p808), DMF is the wrong solvent for hot SPPS for a measurable reason, and the chapter's conclusion that "DMF may not be ideal for standard SPPS steps at high temperature" is better supported by this table than by the synthesis comparisons it rests on.

**Asserted without support — grade D, which is most of the work and is where its value lies.** Aggregation mechanism, aspartimide, DKP, pyroglutamate, β-elimination at C-terminal Cys, multiple addition, Hmb and pseudoproline strategy, the positional 8–12 window, the Pro-is-protective ranking, the 6/12/18/24 spacing rule, chaotropes, capping, DMF quality. The mechanisms and the controlled vocabulary are excellent and I would cite them for mechanism. None of them carries an effect size I would put in a model.

Two D-grade claims deserve singling out as **grade-E risks** if carried forward: the "up to 0.3% per deprotection cycle for Asp-Gly" figure (A p749) rests on a single citation and is not reproduced; and B p667's one-sentence "These findings also hold for solid-phase synthesis reactions" transfers a solution-phase UNCA kinetic result to SPPS with nothing behind it. The racemization numbers are grade A *for solution-phase UNCAs in toluene at 20 °C* and grade E the moment they are quoted as SPPS values. (My own arithmetic, flagged as mine: 0.3%/cycle compounded over 25 deprotections is ~7% cumulative loss — real, but an order of magnitude short of the "difficult sequence" failures the same chapter describes.)

**Contradicted — and all the contradictions are internal to this one work.** This is the review's main result and it is a direct consequence of the skill's step 5.

1. **Most racemization-prone residue.** Stewart: "Of all the natural amino acids, phenylalanine is most susceptible to racemization" (A p727, grade D). Romoff: Phe moderate, Ser/Cys/Asp high (B p667, grade A). **Resolve in favour of the measurement.** The skill's catalogue sides with neither — it names His and Cys and omits Ser and Asp.
2. **Solvent effect on racemization.** Stewart: activated amino acids "racemize more rapidly in polar solvents, such as DMF, than they do in nonpolar solvents, such as CH₂Cl₂" (A p727). Romoff: the rate in CH₂Cl₂ "is comparable to the rate in the polar solvent DMF" (B p667) — *t*½ 23 min vs 13 min, against toluene 132 min and THF 157 min — because tertiary-amine hydrochlorides are soluble in DCM and insoluble in the other two, so dielectric constant is the wrong predictor. **Resolve in favour of the measurement**, and note that a descriptor-based process model using solvent polarity would get this backwards.
3. **Does heat beat aggregation?** Hedged and asserted three pages apart in the same chapter (A p807 vs A p810), with prior literature described as conflicting (A p806). **Unresolved; record as unresolved.**
4. **Chaotropes vs heat.** Effective (A p727, p736) vs "a minor effect" against heating's "more clearly defined improvement" (A p746). Note these are measured on *different steps* — coupling in the Boc chapter, Fmoc removal in the Fmoc chapter — which may dissolve the conflict and is itself the finding.
5. **Where the losses are.** Fmoc chapter: incomplete deprotection, coupling "less common nowadays" (A p749–750). Boc chapter: "either deprotection or coupling may fail" (A p726). PSRG, from measurement: "Side reactions during post-assembly, i.e. cleavage and deprotection, were more significant than actual peptide assembly" (B p766). **All three can hold** — they are about different chemistries and different stages — but anyone building a synthesizability model on assembly-step features alone is, on the only empirical evidence in this source, modelling the wrong stage.
6. **Length.** HW: "The synthesis will rarely collapse after 25 residues" (B p66). The skill's catalogue: risk rises with length. The scanner's `agg.length` fired HIGH at 55. **Reconcilable only by separating aggregation risk from cumulative-yield loss**, which the catalogue conflates under one rule.

**What the outcome measures cannot see.** The Kaiser test is blind in exactly the difficult sequences the Boc protocols target, and brown-colours on N-terminal Asn/Asp/Ser (A p737). Aspartimide and β-aspartyl by-products co-elute with target (A p238). Deamidation is invisible to MALDI-TOF and to quadrupole ESI (B p772). AAA and Edman chemistry strip residual protecting groups and make a failed synthesis look clean (B p768). HPLC + MS cannot distinguish a reversed sequence of equal mass, and 7/20 laboratories did not (B p771). Quantitative per-cycle Fmoc release — the one measure that would ground the multiplicative-yield argument — was attempted and abandoned for insufficient accuracy (A p747).

**One correction to the skill's standing language.** The skill instructs that "non-synthesizable peptide" almost always means low crude purity rather than cannot-be-made. For this source that is wrong often enough to matter: DKP formation cleaves the peptide from the resin (A p728, p749), pyroglutamate formation "stops chain growth" (A p729), and trifluoroacetylation gives "a stable trifluoroacetylated chain which can grow no further" (A p729). These are chain-destroying, not purity-reducing. The correction should be conditional on the outcome measure, not applied as a blanket rewording.

---

---

## Part 2 — Feedback on the skill

Specific and critical, as asked. The skill is genuinely good at what it was built for and I used most of it. The failures below are all about one thing: it was written for a primary paper or a dataset, and it treats "textbook" as a degenerate case.

## 1. The biggest problem: "textbook" is treated as monolithic, and this source is not

The per-source playbook says, in full: *"Textbook / reference work — best source for mechanism and for the controlled vocabulary. Usually no primary data. Cite for mechanism, not for effect size."* The evidence ladder reinforces it: *"most textbook and vendor content"* sits at D.

Applied literally to Houben-Weyl E22 this instruction would have **thrown away the two best pieces of evidence in the source**. B 7.4's UNCA kinetics are a grade-A controlled experiment. B 7.8's ABRF round-robins are eight years of interlaboratory data with central characterisation and a stated outcome measure, including one properly blinded test. Both are inside a "textbook".

The cause is a category error: a multi-author reference work is not a source, it is **~60 sources bound together**, each with its own authors, its own outcome measure and its own evidence grade. Chapters in this one book range from D (A 2.2.2, pure mechanism) to A (B 7.4, B 7.8.4.4). I had to grade per chapter, and nothing in the skill told me to — I did it because grading the volume as a unit produced an answer I could see was wrong.

**Fix:** make the unit of grading the chapter/section, not the file. Add to the playbook list: *"Multi-author reference work or handbook — grade each chapter separately; chapters differ by author and by evidence class, and the mechanism chapters and the data chapters may contradict each other. Expect the primary data to be in the analytical/QC chapters, not the synthesis chapters."* That last clause is what actually happened: the best data were in Vol B's **Analytics** part, which I would have deprioritised if I had been reading for synthesis factors only.

## 2. Step 2 is stated as a hard gate and cannot be satisfied as written

*"Before any other reading, find how the source defines success. ... If you cannot find it, say so and stop treating the source's numbers as comparable to anything else."*

For a 2497-page work there is no single outcome definition, so the gate is unsatisfiable as phrased — the literal reading is "stop", which would end the review. I deviated: I located one outcome definition **per chapter that reports numbers** and built the table above. That was clearly the right move and it did reframe everything (it is how I found the Kaiser-test circularity, which is the best thing in the review). But I was improvising.

**Fix:** the gate should be per claim-cluster, not per source. *"Locate the outcome definition for each distinct body of results. A source may have several; tabulate them and never pool across them."* The existing wording also implies the definition is findable in one place ("in a paper it is in Methods") — add the case where it must be assembled from figure captions and protocol steps, which is what I did for A 4.3.4 (the "210 nm" that defines its purity measure is in a figure footnote, not the text).

## 3. The finding record cannot express a factor that helps on one axis and harms on another

`direction: helps | harms | non-monotone | unclear` is one field over one axis. The source's central practical rule needs two directions at once: **Pro, Arg, His, Gln and Asn prevent assembly failure** (B p66) while Arg, His and Gln are *also* named liabilities in the same work (δ-lactam/des-Arg, epimerisation, pyroglutamate and dehydration). There is no way to record "His helps chain assembly and harms chiral purity" in one record, and splitting it into two records loses the fact that they trade off against each other at the same position.

This is not an edge case. It is the normal shape of an SPPS design decision, and it is why the scanner's split into "chain-assembly difficulty" vs "process liabilities" is right while the record format is not.

**Fix:** allow `direction` to be per-axis, or add an explicit `tradeoff` field naming the countervailing finding. The scanner's own two-section output shows the skill already knows these are different questions.

## 4. Positional structure: the skill says position matters, then provides only whole-sequence machinery

The skill is emphatic and correct that *"means and counts discard position"* and *"localisation usually matters more than totals."* But every positional construct it supplies is **translation-invariant** — sliding windows, run lengths, motif presence. The source's two strongest positional claims are **anchored**, and neither can be expressed:

- **Distance from the resin.** "8–12 residues into the synthesis" (B p66); "the zone about 8–10 residues out from the resin" (A p726). Two independent chapters. This is a position indexed from the C-terminus, and it is computable from sequence for free.
- **Distance from a terminus for side reactions.** The same Gln deamidated at 3-from-C and not at 3-from-N (B p772). N-terminal Gln → pyroglutamate; C-terminal Cys ester → β-elimination and cumulative racemization; C-terminal dipeptide → DKP.

The scanner does have terminal rules (`side.pyroglutamate`, `side.diketopiperazine`), so the capability exists — it just is not generalised, and the catalogue's §1.1 signal table contains no anchored entry at all. Add one: *"Hydrophobic/β-branched content localised 8–25 residues from the C-terminal anchor"*, and note under `agg.length` that the source reports aggregation risk as **non-monotone** in length, not increasing.

## 5. `agg.h_stretch` excludes Ala, and the source's canonical failure is poly-Ala

Catalogue §1.1 defines the classic indicator as runs of {I, L, M, V}. The scanner implements that. Decaalanine therefore scores `h_stretch 0`. Houben-Weyl's founding example of a difficult sequence — with NMR line-broadening evidence and a stated critical zone — is a poly-Ala decamer (A p726). The rule misses the textbook case by construction.

I am not claiming Ala belongs in the same set as Ile at the same weight. The mechanism is β-sheet propensity plus poor solvation, and Ala has high β-sheet propensity with low hydrophobicity, so a Kyte-Doolittle or Janin window partly compensates — which is why the scanner gave decaalanine a MEDIUM rather than nothing. But the catalogue should say so explicitly, because as written it reads as a closed list. Suggest: add a β-sheet-propensity run rule distinct from the hydrophobic run rule, and note that the two dissociate for Ala, Thr and Gln.

## 6. `stability.deamidation` is scoped to Asn-Gly and misses the source's measured case

The catalogue defines deamidation as the Asn-Gly hotspot. The source's only *measured* deamidation is a C-terminal Asp-α-amide and a Gln at 3-from-C (B p772). The rule fires on neither. The catalogue is also right that this is a storage liability rather than an SPPS one — but it then says *"do not conflate"*, which under-sells the finding: the ABRF case is the cleanest positional-vs-compositional demonstration in the entire source and it belongs in the review regardless of which axis it sits on.

## 7. Mitigations are listed without their costs

Catalogue §1.1 lists backbone amide protection (Dmb, Hmb) and pseudoprolines as mitigations, flatly. The source is much more careful, and the carefulness is the useful part:

- Coupling **onto** an Hmb-protected residue is prohibitively slow — >24 h with repeat couplings for β-branched Val/Ile/Thr, "except Hmb-Gly" (B p64, p66). So Hmb converts an aggregation problem into a steric coupling problem at the next position, which is the exact failure mode the catalogue flags elsewhere.
- Hmb removal fails near easily protonated residues — the N-terminal amide bond, the N-terminus, and near His (B p66).
- Introducing Hmb requires its hydroxyl to be Fmoc-protected or the activated carboxyl forms an intramolecular phenyl ester (B p64).

Every mitigation in the catalogue should carry a "cost" line. Without it the catalogue reads as a shopping list, and the thing a reviewer most needs to spot in a paper is an author who applied a mitigation and silently paid its price somewhere else.

## 8. There is no axis for the operator, and the source says that is the largest factor

*"peptide quality depends the same or to a greater extent on the experience and skill of personnel who carry out the synthesis than on the instrument or method used"* (B p766), from the body that ran eight interlaboratory studies. Backed by 1996's racemization spread where one laboratory held all the high values.

The three axes have nowhere to put this. It is not sequence-intrinsic, not process (it is *why* process is not reproducible), and not analytical. I recorded it with `axis: NONE OF THE THREE`, which is unsatisfying.

This matters more than a taxonomy quibble. The skill's audit section tells me to stratify by length, source and batch and to look for one stratum dominating. **Laboratory is the stratum this source identifies as dominant**, and "source library" in the audit checklist is not quite the same thing — a pooled synthesizability dataset drawn from several facilities has a lab effect that any sequence feature correlated with facility will absorb. Either add a fourth axis (provenance/operator) or add laboratory explicitly to the cohort-structure checklist with this citation behind it.

## 9. Busywork

Honestly, very little. Two things:

- **The per-source playbook for "textbook" earned nothing** — see §1. Its single actionable instruction ("cite for mechanism, not effect size") would have caused an error here.
- **The "Tools" section was dead weight for this task.** It tells me to check for a literature MCP, web search, chemistry databases. The material was three local PDFs; I needed none of them, and the available ChEMBL/PubMed/ClinicalTrials servers have no bearing on a 1999 synthesis handbook. (Seven bio-research MCP servers also failed to connect with proxy 403s — not the skill's fault, and I did not need them.) The section should say: if the source is in hand, read the source.

Everything else I used. The claims-to-challenge table, the evidence ladder, the three axes, step 5's cross-check instruction and the catalogue's "absence of signal is evidence about the assay" framing all paid off — the cross-check instruction is what produced the six internal contradictions, which are the most useful output of this review.

## 10. What I needed and did not get

- **A quoting protocol for OCR'd scanned text.** The practical notes in the task covered searching (single tokens, which was correct and necessary) but not *quoting*. Reflowed two-column OCR silently reorders lines, so a quotation assembled from adjacent fragments can be wrong in ways that read fluently. The Vol B Table 4 header scramble would have inverted the headline racemization finding if I had trusted the table alone. I worked out a rule — reconstruct, then verify with `grep -F` on a distinctive substring, then cross-check against an independent statement of the same fact — and the skill should carry it, because the skill's own targets (vendor handbooks, old application notes, textbook chapters) are exactly the material that arrives as scans.
- **A place to record conflicts *within* one source.** `conflicts: other sources that disagree` assumes conflicts are between sources. Six of my seven substantive contradictions were between chapters of one book, and they were the most informative findings. Reword to "other sources **or sections** that disagree", and add to step 5: *"check whether the source contradicts itself; in multi-author works, expect the mechanism chapters and the data chapters to disagree, and prefer the measurement."*
- **Guidance on what to do when calibration mode is unavailable.** The calibration section assumes labelled outcomes. A textbook has sequences and prose but no label column, so `--label-column` was unusable and I got flag mode only. The skill's honest-limits section says the flags are uncalibrated — fine — but it offers no substitute check. What I actually did, and what the skill should recommend, is **run the scanner on the source's own named failure cases and see whether it flags them.** That is a free validity check against a ground truth the source hands you, and it is how I found the decaalanine and TRNLADQED gaps. It is the nearest thing to calibration available for a non-dataset source, and it found two real rule defects in one run.
- **A `confidence` or `reconstruction` field on the finding record** for exactly this kind of source, distinguishing "quoted verbatim" from "reassembled from OCR fragments". I ended up carrying that distinction in prose.

## 11. Things in the material the skill gave me no way to record

- **Resin shrinkage during synthesis as a free live diagnostic.** "a characteristic shrinkage of the peptide-resin during synthesis" (B p64); "severe shrinkage of the resin" (A p726). No catalogue entry, and it is a zero-cost observable.
- **Dibenzofulvene release peak shape.** A sharp peak means rapid release; "a broad peak with extending tail is associated with the onset of aggregation problems within the matrix" (A p747). The catalogue's Axis 3 table lists "Quantitative Fmoc release" as a per-cycle measure, but this is a *qualitative shape* read — a different and more available thing.
- **Multiple addition / double insertion.** Premature Fmoc removal during coupling under TBTU/DIPEA inserts a residue twice (A p750). A +1-residue by-product, and the catalogue covers deletion and truncation but not insertion.
- **β-elimination at C-terminal Cys esters** → dehydroalanine → piperidine adduct, "virtually avoided" with Trt but not Acm side-chain protection (A p748). A residue × position × protecting-group interaction with no catalogue entry.
- **Dehydration of Gln and Asn** on prolonged activation (A p811), with the practical consequence that you should recouple rather than extend coupling time — which cuts directly against the catalogue's framing of extended coupling as "the standard lever for difficult positions."
- **Trifluoroacetylation as irreversible chain termination** from residual TFA activated by diimide, and quaternary ammonium groups on chloromethyl resins that bind TFA strongly (A p729). The catalogue has guanidination under §1.7 "irreversible chain termination" but not this, and the §1.7 framing ("not sequence-detectable — it is a stoichiometry failure") should be widened: there is a family of irreversible terminations with different causes.
- **Segment-condensation rules.** The C-terminal residue of an activated segment is "classically limited to glycine ... and proline" (B p66, A p803), and segment-level deletions are *easier* to purify than single-residue deletions (B p65). The skill has one line on per-coupling vs whole-sequence modelling but no vocabulary for convergent synthesis, where the governing constraints are different.
- **A negative result.** "No correlation was found to exist between coupling time and percent correct product" across 33 laboratories (B p771). The skill's claims-to-challenge table is built to deflate positive claims and has no slot for a reported null — which is a shame, because a null on the catalogue's "standard lever" is worth more than most positives.

## 12. One factual correction to the skill's content

Catalogue §1.4 names the residues most prone to racemization as **His, Cys, and N-alkylated amino acids**. The only controlled kinetic dataset in this source (B p665–667) puts **Ser highest**, then Cys and Asp, with Phe and Ala moderate and Val/Ile low — Ser and Asp being absent from the catalogue entirely. The mechanism given (β-electron-withdrawing substituent raises propensity; β-branching lowers it) predicts the whole ordering, including why His belongs there and why Val does not. The catalogue's list looks like it was assembled from which residues get special protecting groups in practice rather than from measured rates. Worth reconciling, with the mechanism stated — it generalises to non-standard residues, whereas a list of three does not.

Separately, §1.2's mitigation list (OMpe, OEpe, OPhp, OBno) is all post-1999 and finds no counterpart here; this source gives cyclohexyl ("especially" efficient), fluorenylmethyl and allyl esters (A p239). Not an error, but the catalogue would be more useful to a reviewer of older material if its vendor-era entries were dated.
