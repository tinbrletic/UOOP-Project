# SPPS liability rules — mechanism, source, and measured performance

Each rule implemented in `../scripts/scan_peptides.py`, with its mechanism, its source, and
how it performed against 1771 labelled syntheses.

## Sources

| Tag | Source |
|---|---|
| **HW-E22a** | Houben-Weyl, *Methods in Organic Chemistry: Synthesis of Peptides*, Vol. E22a (1999), §1.2.1.3 "Difficult Sequences". In repo: `docs/Marta de Zotti - materials/HOUBEN WEYL E22A.pdf` |
| **GPT** | Gyros Protein Technologies, *SPPS Tips For Success: A Guide to Better Peptide Synthesis*. In repo: `docs/SPPS Tips For Success Handout.pdf` |
| **Gutman-2022** | Gutman et al., *Predicting the Success of Fmoc-Based Peptide Synthesis*, ACS Omega 2022. In repo: `docs/paper.pdf`; feature benchmark in `data/raw/ao2c02425_si_002.xlsx` sheet S3 |
| **general** | Standard peptide-chemistry knowledge, not traceable to the above. Lower confidence; flagged as such in output. |

---

## Category A — Chain-assembly difficulty

> HW-E22a §1.2.1.3: "the most serious potential problem facing both synthesis in solution and
> on solid support is the sequence-dependent poor solubility or solvation of the protected
> peptides, which prevents reactions from proceeding to completion. In solution and on resin,
> this phenomenon, described as **difficult sequences**, derives from the intermolecular
> association of the protected peptide derivatives into β-sheet-like structures... In these
> aggregates the N-termini become inaccessible for further chemistry."

Documented remedies: solvent changes ("magic mixtures"), chaotropic salts, elevated
temperature, solubilising protecting groups, backbone amide protection, and **pseudoprolines**
— which are installed at **Ser, Thr or Cys**.

| Rule | Mechanism | Source | Lift | Within-cohort AUC |
|---|---|---|---|---|
| `agg.hydrophobic_window` | Max mean Kyte-Doolittle over a 5-residue window. Localised hydrophobicity drives β-sheet association. | HW-E22a | **3.19** | 0.697 |
| `agg.h_stretch` | Longest consecutive run of I/L/M/V. The paper's `h stretch`. | Gutman-2022, HW-E22a | 2.16 | 0.581 |
| `agg.beta_branched_run` | Run of ≥3 β-branched residues (I/V/T). Steric hindrance slows coupling; GPT shows hindered couplings needing 2×30 min to reach 74% crude purity vs 45% at 2×3 min. | GPT | 1.93 | — |
| `agg.length` | Multiplicative yield loss with chain length. | Gutman-2022 | 0.52 | **0.500 — confounded** |
| `agg.pseudoproline_site` | Ser/Thr/Cys in or beside a difficult run = a site where a pseudoproline dipeptide can disrupt the β-sheet. Reported as an **opportunity**, not a liability. | HW-E22a, GPT | — | — |

**Note on `h_stretch`:** mechanistically the most satisfying feature, but empirically weaker
than a plain mean hydrophobicity (0.581 vs 0.740 within-cohort). Report it; do not lead with
it. Compositional counts of aromatic, polar and acidic residues score AUC 0.48–0.51 in
Gutman-2022's own benchmark — composition alone is not the mechanism.

---

## Category B — Process liabilities

Real chemistry with specific mitigations, but **not predictive of crude purity** on an
MS-abundance label. Use as a protocol checklist.

### Aspartimide — `side.aspartimide`

Repeated piperidine exposure on Asp-containing sequences gives aspartimide and up to nine
by-products. Severity depends on the residue **following** Asp. Graded table from GPT:

| Asp-Xaa | Sensitivity | Rule severity |
|---|---|---|
| Asp-**Gly** | +++++ | CRITICAL |
| Asp-Asn | +++ | HIGH |
| Asp-Asp, Asp-Ser, Asp-Thr, Asp-Cys | ++ | MEDIUM |
| Asp-Ala, Asp-Gln, Asp-His (pH dependent) | + | LOW |

Also influenced by base, temperature, and solvent (DMSO > DMF >> THF > DCM). Mitigations:
piperazine with 0.1 M HOBt instead of piperidine; Asp(OMpe)/Asp(OEpe) side-chain protection;
backbone protection (Dmb/Hmb); pseudoprolines.

**Measured lift 1.37; `has_DG` as a standalone feature scores AUC 0.505 — noise.** The
chemistry is real and well-documented; this particular label cannot see it. Flag it for
protocol design, not for risk ranking.

### Diketopiperazine — `side.diketopiperazine`

GPT: "one of the most prevalent side reactions in SPPS" — the N-terminus attacks the
C-terminal residue, forming a 6-membered ring and **cleaving the peptide off the resin**.
Exacerbated by N-Me-Xaa, alternating D/L residues, and **Pro at the C-terminus**. Most likely
during Fmoc deprotection. Mitigations: shorter deprotection, 0.1 M HOBt/Oxyma in the
piperidine, 2-chlorotrityl resin when Pro is C-terminal, carbodiimide coupling.

Detectable from sequence: Pro or Gly in the C-terminal dipeptide. Lift 1.46.

### Pyroglutamate — `side.pyroglutamate`

GPT: an N-terminal Gln cyclises to pyroglutamate, capping the chain. Detectable: Gln at
position 1 (HIGH); Glu at position 1 (LOW). Lift 1.16.

### Racemization — `side.racemization`

GPT: residues prone to epimerisation are **His, Cys**, and N-alkylated amino acids, via the
oxazolone mechanism. Worse with base-mediated activation, pre-activation and heating.
Mitigations: carbodiimide activation, HCTU or COMU, 2,4,6-collidine instead of DIPEA/NMM,
His(Mbom) instead of His(Trt).

**Lift 0.91 — no discriminative power.** Keep as a protocol note.

### Arg δ-lactam — `side.arg_lactam`

GPT: δ-lactam forms during Arg activation, creating substoichiometric quantities and causing
**des-Arg deletion peptides**; recoupling is commonly required. Worse under microwave.

**Lift 0.58 — fires more often on SUCCESSES.** This is the ESI ionisation confound: Arg is
basic and raises MS response, inflating the apparent abundance ratio. Emitted at INFO with an
explicit warning. **Never use Arg content as a difficulty signal on an MS-derived label.**

### Disulfides and oxidation — `side.disulfide`, `side.oxidation`

≥2 Cys requires orthogonal protection and a defined oxidation order. Met/Trp/Cys are
oxidation-sensitive and drive cleavage-cocktail scavenger choice. `side.oxidation` fires on
~54% of a typical set at lift 1.15 — near-universal and non-discriminative, so emitted at
INFO.

### Deamidation — `stability.deamidation`

Asn-Gly is a deamidation hotspot. A **storage/stability** liability, not an SPPS one.
Source: **general**. Lift 0.39. Emitted at LOW and labelled as such.

---

## Also documented in the sources, not sequence-detectable

Listed so an agent does not mistakenly claim the scanner covers them:

- **Guanidination** — excess HATU relative to amino acid guanidinates the free amine and
  terminates the chain irreversibly. Fixed by stoichiometry (e.g. 0.95 eq HATU per 1 eq AA)
  or pre-activation. Depends on process, not sequence. (GPT)
- **`left drop` / `all drop` / `right drop` / `all bond`** — Gutman-2022 features derived from
  observed MS evidence of single-residue loss. These are **measured from spectra, not computed
  from sequence**, so they cannot be used to score an unsynthesised peptide. `right drop`
  scored AUC 0.482 in the paper's own benchmark.

---

## Validation method

`lift` = P(failure | rule fires) / P(failure | rule does not fire), on
`data/raw/peptide_baza.csv` (n=1771, 232 failures). `AUC` = rank-based AUC for predicting
failure. "Within-cohort" = computed inside the length-15 stratum (n=1036, 202 failures), which
removes the cohort confound described in SKILL.md.

Reproduce with:

```bash
python scripts/scan_peptides.py data/raw/peptide_baza.csv --format csv > /tmp/scan.csv
```

then join on the label column and recompute. Any re-calibration on a new dataset should
repeat the stratified check before the numbers in these tables are trusted.
