# Catalogue of factors affecting peptide synthesis outcome

A lookup table for recognising a factor when a source names it — often obliquely, and rarely
with the same vocabulary twice. Organised by the three axes in SKILL.md.

Each entry gives the **mechanism**, the **vocabulary** sources use for it, and whether it is
**sequence-detectable** (computable before the peptide exists) or only observable after the
fact. `→ rule` names the corresponding check in `../scripts/scan_peptides.py` where one exists.

**Source types** used below: `textbook` (reference works and monographs on peptide synthesis),
`vendor` (synthesiser and reagent manufacturers' handbooks and application notes), `primary`
(peer-reviewed studies). Most mechanism is textbook-grade; most graded motif tables are
vendor-grade. Neither is a substitute for a controlled experiment — see the evidence ladder in
SKILL.md.

---

## Axis 1 — Sequence-intrinsic factors

### 1.1 On-resin aggregation ("difficult sequences") — the dominant mechanism

**Mechanism.** As the protected chain grows, sequence-dependent poor solvation drives
intermolecular association into β-sheet-like structures. In these aggregates the reactive
N-termini become inaccessible, so deprotection and coupling cannot go to completion. This is
a *solvation* failure, not a reactivity failure — which is why the remedies are mostly about
disrupting the structure rather than activating harder. (`textbook`)

**Vocabulary.** "difficult sequence", "difficult coupling", "aggregation", "β-sheet
formation", "poor solvation", "incomplete coupling", "chain association", "onset of
aggregation", sometimes "sequence-dependent synthesis failure".

**Sequence-detectable signals:**

| Signal | Note | → rule |
|---|---|---|
| Runs of consecutive aliphatic-hydrophobic residues (I, L, M, V) | The classic indicator. Run length, not total count. | `agg.h_stretch` |
| Localised hydrophobicity over a sliding window | Often a better signal than a whole-sequence mean, because it preserves position | `agg.hydrophobic_window` |
| Runs of β-branched residues (I, V, T) | Steric hindrance compounds the solvation problem | `agg.beta_branched_run` |
| Chain length | Risk rises with length in principle, but **length frequently acts as a cohort proxy in real datasets** — stratify before believing it | `agg.length` |
| Low Gly/Pro content in a hydrophobic region | Gly and Pro disrupt β-sheet; their absence permits it | — |

**Hydrophobicity scales.** Sources use Janin, Kyte-Doolittle, Rose, Eisenberg, Engelman GES,
Hopp-Woods, Cornette and others. They correlate strongly; a result that holds on one usually
holds on several. If a source reports only one scale and a striking result, check whether it
is scale-specific — that would be a red flag.

**Mitigations a source may describe** (their presence tells you the author expected
aggregation): solvent changes and "magic mixtures"; chaotropic salt additives; elevated
temperature; backbone amide protection (Dmb, Hmb); **pseudoproline dipeptides**; lower resin
loading; solubilising protecting groups; fragment condensation or ligation instead of linear
assembly.

> **Pseudoprolines are derived from Ser, Thr and Cys.** So a Ser/Thr/Cys inside or beside a
> hydrophobic run is an *opportunity*, not a liability — a site where the β-sheet can be
> disrupted by design. This is why composition counts of polar residues often show no signal
> while the mechanism is nonetheless real: position relative to the difficult region is what
> matters. → `agg.pseudoproline_site`

### 1.2 Aspartimide formation

**Mechanism.** Repeated base exposure (piperidine, each deprotection cycle) on Asp-containing
sequences promotes cyclisation to a succinimide, which then opens to a family of by-products
including the β-peptide and the epimer — reported as up to nine distinct species. Risk
depends strongly on the residue **following** Asp. (`vendor`, `primary`)

**Vocabulary.** "aspartimide", "succinimide", "α/β-peptide", "piperidide", "[M+67]",
"base-mediated side reaction".

**Sequence-detectable.** Asp-Xaa dipeptide motif. Published graded sensitivity (`vendor`),
highest first: **Asp-Gly** ≫ Asp-Asn > Asp-Asp ≈ Asp-Ser ≈ Asp-Thr ≈ Asp-Cys > Asp-Ala ≈
Asp-Gln ≈ Asp-His (pH-dependent). → `side.aspartimide`

**Also modulated by** (process axis): base identity, temperature, solvent (DMSO > DMF ≫ THF >
DCM), microwave, and the Asp side-chain protecting group — OtBu is standard and more
susceptible; OMpe, OEpe, OPhp, OBno are mitigations. Backbone protection and pseudoprolines
also suppress it.

**Note on detectability by assay.** Several aspartimide by-products are isobaric or
near-isobaric with the target. Whether this liability appears in a dataset depends entirely
on whether the analytical method resolves them. Absence of signal is evidence about the
assay.

### 1.3 Diketopiperazine (DKP) formation

**Mechanism.** The free N-terminus of the resin-bound dipeptide attacks the C-terminal ester,
forming a six-membered ring and **cleaving the peptide off the resin** — an outright loss, not
an impurity. Occurs mainly during deprotection of the second residue. Accelerated when the
two residues can adopt a cis arrangement. (`vendor`, `textbook`)

**Vocabulary.** "diketopiperazine", "DKP", "2,5-diketopiperazine", "chain loss from resin",
"first-cycle loss".

**Sequence-detectable.** Risk motifs at the C-terminal dipeptide: **Pro at position 1 or 2
from the C-terminus**, N-methylated residues, alternating D/L configuration, and Gly in the
C-terminal dipeptide. → `side.diketopiperazine`

**Mitigations.** 2-chlorotrityl (CTC) resin when Pro is C-terminal (steric block); shorter
deprotection; HOBt or Oxyma added to the piperidine; carbodiimide coupling to avoid tertiary
base.

### 1.4 Racemization / epimerisation

**Mechanism.** Base-mediated activation can form an oxazolone intermediate at the activated
residue, which racemises. Worst with pre-activation, strong base and heat. (`vendor`,
`textbook`)

**Vocabulary.** "racemization", "epimerization", "oxazolone", "D-isomer", "diastereomer",
"chiral purity".

**Sequence-detectable.** Residues reported as most prone: **His, Cys**, and N-alkylated amino
acids. Cys protecting groups differ substantially in how much they suppress it. → `side.racemization`

**Mitigations.** Carbodiimide activation; HCTU or COMU; weaker/hindered base such as
2,4,6-collidine in place of DIPEA or NMM; alternative side-chain protection (e.g. Mbom for
His).

### 1.5 N-terminal and residue-specific liabilities

| Liability | Mechanism | Detect | → rule |
|---|---|---|---|
| **Pyroglutamate** | N-terminal Gln cyclises, capping the chain | Gln at position 1 (Glu lower risk) | `side.pyroglutamate` |
| **Arg δ-lactam** | Arg side chain cyclises during activation, giving substoichiometric coupling and **des-Arg deletion peptides**; recoupling often needed; worse under microwave | Arg present | `side.arg_lactam` |
| **Disulfide strategy** | ≥2 Cys requires orthogonal protection and a defined oxidation order | Cys count ≥ 2 | `side.disulfide` |
| **Oxidation** | Met, Trp, Cys are oxidation-sensitive; drives cleavage-cocktail scavenger choice | Met/Trp/Cys present | `side.oxidation` |
| **Deamidation** | Asn-Gly is a deamidation hotspot — a **storage/stability** liability, not an SPPS one; do not conflate | Asn-Gly motif | `stability.deamidation` |
| **Steric hindrance** | α,α-disubstituted residues (e.g. Aib) and N-methylated residues couple slowly; vendor data show crude purity roughly doubling when coupling time is extended at the hindered position | Non-standard residues; β-branched runs | `agg.beta_branched_run` |

### 1.6 Deletion vs truncation — a distinction sources blur

- **Deletion**: one residue missing, chain continued. Nearly identical to the target and hard
  to separate.
- **Truncation**: chain stopped. Easier to separate, and **capping** (e.g. acetic anhydride)
  deliberately converts deletions into truncations for this reason.

A source reporting "deletion sequences" without saying whether capping was used is describing
a different process from one that caps. Record it.

### 1.7 Irreversible chain termination

**Guanidination.** Excess uronium/aminium coupling reagent (e.g. HATU) relative to amino acid
can guanidinate the free N-terminal amine. The product is stable through the rest of the
synthesis and to TFA cleavage, so it **permanently terminates** that chain. Fixed by
stoichiometry (reagent slightly sub-stoichiometric to amino acid) or by pre-activation —
though pre-activation raises racemization risk. (`vendor`)

**Not sequence-detectable** — it is a stoichiometry failure. Listed so you do not mistake a
process problem for a sequence problem.

---

## Axis 2 — Process and environment factors

These are what a source must disclose for its sequence-level claims to be interpretable. When
reviewing, **record which of these the source states and which it omits.**

| Factor | Why it matters | What to extract |
|---|---|---|
| **Resin and linker** | Determines C-terminal functionality (acid vs amide), DKP susceptibility, and swelling. CTC resists DKP; Wang, Rink amide behave differently | Resin type, linker, substitution |
| **Resin loading** | Lower loading increases inter-chain spacing and reduces aggregation. A frequently decisive variable | mmol/g |
| **Coupling reagent** | Carbodiimide (DIC + HOBt/Oxyma), aminium/uronium (HBTU, HATU, HCTU, COMU), phosphonium (PyBOP, PyOxim) differ in efficiency, racemization suppression and guanidination risk | Reagent, equivalents |
| **Base** | DIPEA, NMM, 2,4,6-collidine — directly modulates racemization and aspartimide | Identity, equivalents |
| **Solvent** | DMF, NMP, DCM, DMSO and mixtures. Affects solvation and aspartimide rate. Secondary-amine impurities in aged DMF cause premature deprotection — a documented cause of irreproducibility | Solvent, grade, additives |
| **Temperature / microwave** | Speeds coupling and can overcome aggregation, but increases racemization, aspartimide and Arg lactam | Temperature, heating method |
| **Coupling time, excess, double coupling** | The standard lever for difficult positions; vendor data show large crude-purity gains from extended coupling at hindered residues | Time, equivalents, single/double |
| **Deprotection regime** | Piperidine concentration and time; piperazine as an alternative; HOBt/Oxyma additives. Shorter is better for DKP and aspartimide | Reagent, concentration, duration |
| **Capping** | Converts deletions to truncations | Whether used, reagent |
| **Backbone protection / pseudoprolines** | Direct aggregation mitigation | Positions used |
| **Cleavage cocktail and scavengers** | TFA with TIS, EDT, water, thioanisole etc. Determines oxidation and alkylation side products — a source of apparent "synthesis failure" that is really a cleavage artefact | Composition, time |
| **Synthesis mode** | Batch vs flow vs microwave-assisted; manual vs automated | Platform |

> **Per-coupling vs whole-sequence modelling.** The strongest published synthesis models take
> the pre-synthesised chain, the incoming monomer and the process conditions, and predict one
> coupling step. Whole-sequence aggregate descriptors throw that structure away. When
> reviewing a model, note which framing it uses; it bounds what the model can represent.

---

## Axis 3 — Analytical and measurement factors

**Determine this first.** It bounds every conclusion a source can support.

| Outcome measure | What it actually reports | Known biases |
|---|---|---|
| **Crude HPLC purity (UV)** | Fraction of UV absorbance under the target peak | Detector response is not uniform across species; co-elution of close analogues (deletions, epimers) inflates apparent purity |
| **MS peak abundance ratio** | Target-matching m/z intensity over total | **Ionisation efficiency is strongly sequence-dependent** — basic residues (H, K, R) ionise well in positive ESI and inflate apparent purity. Isobaric by-products are invisible |
| **Isolated yield** | Mass recovered after purification | Confounds synthesis quality with purification losses and operator effort |
| **Resin colour tests** (Kaiser/ninhydrin, chloranil) | Free amine present after coupling | Qualitative; unreliable for secondary amines such as Pro |
| **Quantitative Fmoc release** | Per-cycle deprotection efficiency | Among the better per-cycle measures; rarely reported in datasets |

**Thresholding.** Many datasets binarise a continuous purity measure at an arbitrary cut
(50% is common). Two near-identical peptides either side of the cut land in opposite classes.
Always record the threshold and whether the continuous value is recoverable.

**The practical consequence.** A liability is only learnable from a dataset if the assay can
see it. When a well-established chemical liability shows no statistical signal, the ordered
explanations are: (1) the assay cannot resolve it; (2) the protocol already suppressed it;
(3) it is genuinely rare in that sequence space. Only after excluding those is "the chemistry
is wrong" on the table — and it is rarely the answer.

---

## Reporting template

```markdown
## Source
<citation>  type: primary|vendor|textbook  outcome measure: <...>  threshold: <...>

## Factors claimed
| factor | axis | direction | mechanism | evidence | scope | detectable |

## Process disclosed / omitted
<resin, reagent, base, solvent, temperature, deprotection, capping, cleavage>

## Assessment
- supported:
- asserted without support:
- contradicted by:
- what the outcome measure cannot see:
```
