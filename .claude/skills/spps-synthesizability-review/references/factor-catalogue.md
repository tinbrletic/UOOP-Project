# Catalogue of factors affecting peptide synthesis outcome

A lookup table for recognising a factor when a source names it — often obliquely, and rarely
with the same vocabulary twice. Organised by the four axes in SKILL.md.

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

Aggregation needs **two** things together: a backbone free to adopt regular extended
structure, **and** poor solvation. Either alone is insufficient — which is why a
disruptor-free run of charged residues does not aggregate, and why the hydrophobic-run
signal alone misses the canonical case.

| Signal | Note | → rule |
|---|---|---|
| Runs of consecutive aliphatic-hydrophobic residues (I, L, M, V) | The classic indicator, but **not a closed list** — see the poly-Ala note below. Run length, not total count. | `agg.h_stretch` |
| Localised hydrophobicity over a sliding window | Better than a whole-sequence mean, because it preserves position | `agg.hydrophobic_window` |
| Runs of β-branched residues (I, V, T) | Steric hindrance compounds the solvation problem | `agg.beta_branched_run` |
| **Homopolymer runs** | Maximal backbone regularity, no disruption. Catches poly-Ala, which the I/L/M/V rule scores zero. | `agg.homopolymer_run` |
| **Stretches containing no Gly or Pro** | Gly and Pro are the two β-sheet disruptors: Pro cannot donate a backbone NH and is conformationally restricted, Gly is entropically costly to order. Gate on charge density — a charged disruptor-free stretch is soluble. | `agg.no_disruptor_window` |
| **β-sheet propensity over a window** | Independent of hydrophobicity; the two **dissociate** for Ala, Thr and Gln | `agg.sheet_propensity_window` |
| **Hydrophobic/β-branched content 8–25 residues from the C-terminal anchor** | The reported collapse window. **Anchored, not translation-invariant** — see below. | `agg.anchor_zone` |
| Chain length | **This is cumulative-yield risk, not aggregation risk.** Kept as a separate rule for that reason. Length also frequently acts as a cohort proxy in real datasets — stratify before believing it. | `agg.length` (category `YIELD`) |

> **The poly-Ala correction.** The founding example of a difficult sequence in the primary
> reference literature is **decaalanine**, which failed on standard polystyrene with
> β-structure, severe resin shrinkage, and NMR line-broadening as physical evidence. Alanine
> is in none of the hydrophobic sets, and its own β-sheet propensity is *low*
> (Chou-Fasman P<sub>β</sub> ≈ 0.83). It aggregates because a homopolymer of a small,
> unhindered residue gives maximal backbone regularity with no disruptor and no steric
> relief. **Any rule set built only on hydrophobic-residue identity misses the textbook case
> by construction.** Encode regularity as well as hydrophobicity.

> **Position is anchored, not relative.** The literature reports aggregation onset as a
> *window measured from the resin* — "the critical region was the zone about 8-10 residues
> out from the resin" — and reports that synthesis "will rarely collapse after 25 residues".
> So **aggregation risk is non-monotone in chain length**: it rises into a window and then
> falls. Sliding windows, run lengths and motif presence are all translation-invariant and
> cannot express this. Index from the C-terminus. This also means a flat "longer is riskier"
> rule conflates two different mechanisms — aggregation (non-monotone, windowed) and
> cumulative stepwise yield loss (monotone, multiplicative). Keep them apart.

**Hydrophobicity scales.** Sources use Janin, Kyte-Doolittle, Rose, Eisenberg, Engelman GES,
Hopp-Woods, Cornette and others. They correlate strongly; a result that holds on one usually
holds on several. If a source reports only one scale and a striking result, check whether it
is scale-specific — that would be a red flag.

**Mitigations a source may describe** — their presence tells you the author expected
aggregation. **Every one has a cost, and the cost is the part a reviewer needs**, because
the common failure is an author who applied a mitigation and silently paid its price
somewhere else.

| Mitigation | Cost / caveat |
|---|---|
| Solvent change, "magic mixtures" | Changes racemization and aspartimide rates at the same time; a solvent swap is never a single-variable change |
| Chaotropic salts (NaClO₄ > KSCN) | **Contested.** Presented as the remedy in some chapters and as "a minor effect" versus heating in others — note that these are often measured on *different steps* (coupling vs Fmoc removal), which may dissolve the conflict |
| Elevated temperature (~60 °C) | Raises racemization, aspartimide and Arg δ-lactam. Benefit against aggregation is **explicitly uncertain** in the source literature, hedged and asserted within the same chapter |
| Backbone amide protection (Hmb, Dmb) | **Costs a coupling.** Acylating *onto* an Hmb-protected residue is prohibitively slow for β-branched Val/Ile/Thr — >24 h with repeat couplings, except Hmb-Gly. Removal fails near easily protonated residues and near His. Introduction needs the Hmb hydroxyl protected or the activated carboxyl forms an intramolecular phenyl ester. So Hmb converts an aggregation problem into a steric coupling problem one position later |
| Pseudoproline dipeptides | Only available at Ser, Thr or Cys positions; reverts to the native residue on cleavage, so it constrains where you can intervene, not whether |
| Lower resin loading | Throughput cost; reported ranges differ by an order of magnitude between contexts (<0.15 mmol/g, 0.04–0.3 mmol/g, ~0.05 mmol/g) |
| Fragment condensation / ligation | The C-terminal residue of an activated segment is classically limited to **Gly or Pro** to avoid epimerising the junction. Upside: segment-level deletions differ by a whole segment and are *easier* to purify than single-residue deletions |
| Capping (Ac₂O, acetylimidazole) | Does not improve yield — it converts deletions into truncations, which are easier to separate. A purification aid, not a synthesis fix |
| Extended coupling time / more equivalents | Usually the first lever reached for, but **see the caveats under Axis 2** — prolonged activation dehydrates Gln and Asn, and one interlaboratory study found no correlation between coupling time and correct product |

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
DCM), microwave, and the Asp side-chain protecting group. Backbone protection and
pseudoprolines also suppress it.

**Protecting-group mitigations are era-dependent** — date them when reviewing older material:

| Era | Asp side-chain protection offered |
|---|---|
| Late-1990s reference literature | cyclohexyl (described as "especially" efficient), fluorenylmethyl, allyl esters |
| Current vendor literature | OMpe, OEpe, OPhp, OBno; OtBu as the standard and more susceptible option |

A reviewer of pre-2000 material will not find the modern esters, and their absence is not
evidence the authors ignored the problem.

**Graded tables are not universal either.** The nine-way Asp-Xaa ranking above comes from
vendor literature. The principal reference work has **no such table** — it names
Asp-Gly/Asp-Ser/Asp-His in one chapter and Asp-Asn/Asp-Gly/Asp-Gln/Asp-Ser in another,
ungraded, and puts Asp-Gln in its "especially susceptible" list where the vendor table puts
it near the bottom. Treat the grading as one source's ordering, not a constant.

**A quantitative figure to handle carefully.** "Up to 0.3% per deprotection cycle" for
Asp-Gly appears in the reference literature on a single citation and is not reproduced there.
Compounded over 25 deprotections that is ~7% cumulative — real, but an order of magnitude
short of the difficult-sequence failures described in the same chapter. Quote it as a single
unreplicated figure.

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

**Sequence-detectable.** Two orderings are in circulation and they disagree. Prefer the
measured one.

**Measured (controlled kinetics, chiral HPLC half-lives, one variable at a time):**

| Propensity | Residues |
|---|---|
| **High** | **Ser, Cys, Asp** |
| Moderate | Phe, Ala |
| Low | Val, Ile |

The **mechanism predicts the whole ordering**, which makes it extensible to non-standard
residues in a way a list of names is not:

- *thermodynamic* — a **β-electron-withdrawing** side-chain group stabilises the α-carbanion
  and **raises** propensity (hence Ser, Cys, Asp — OH, SH, COOH at the β position)
- *kinetic* — **β-branching** sterically blocks base approach to the α-proton and **lowers**
  it (hence Val, Ile, Thr)

Reported half-lives span roughly 5000× across the series, so the *ordering* is robust even
though replication is not stated.

> **Earlier versions of this catalogue listed "His, Cys and N-alkylated" as the most prone.**
> That list omits Ser and Asp, and Ser is the worst residue in the only controlled dataset
> available. The old list was assembled from *which residues get special protecting groups
> in practice* — His(Mbom/Bum), Cys(Trt) — rather than from measured rates. Those protecting
> groups exist for good reasons, but prevalence of a workaround is not the same quantity as
> rate. His does belong in the high group on mechanism; Ala and Phe do not.

**Scope caveat that matters.** The controlled data were obtained in **solution phase**, on
urethane-protected N-carboxyanhydrides, in toluene at 20 °C. Transfer to SPPS is *asserted*
in the source in a single sentence with no supporting data. Quote these as solution-phase
kinetics; quoting them as SPPS values is a grade-E move. → `side.racemization`

**A contradiction to expect.** Older chapters of the same reference work state that
phenylalanine is the most racemization-prone natural amino acid, and that the rate is higher
in DMF than in DCM. Both are contradicted by the measured data in the same work — DCM's rate
is comparable to DMF's, because tertiary-amine hydrochlorides are soluble in DCM and
insoluble in toluene and THF, so **dielectric constant is the wrong predictor**. A
descriptor-based process model using solvent polarity will get this backwards.

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
| **Deamidation** | Asn-Gly/Asn-Ser are motif hotspots, but deamidation is also **positional**: the same Gln has been reported to deamidate at 3 residues from the C-terminus and *not* at 3 from the N-terminus. A C-terminal Asx-α-amide is especially susceptible. Primarily a **storage/stability** liability rather than an SPPS one — but it is the cleanest published demonstration of *localisation beating composition*, so record it whichever axis it sits on | Asn-Gly/Asn-Ser motif; Asn/Gln within 3 of the C-terminus | `stability.deamidation`, `stability.deamidation_cterm` |
| **β-elimination at a C-terminal Cys ester** | Eliminates to dehydroalanine, which then forms a piperidine adduct. A **residue × position × protecting-group** interaction: virtually avoided with Trt side-chain protection, *not* avoided with Acm. The same C-terminal Cys ester also racemizes cumulatively with the number of Fmoc-removal cycles, so risk scales with chain length | Cys at the C-terminus | — |
| **Dehydration of Gln and Asn** | Prolonged *activation* dehydrates the side-chain amide to a nitrile. Practical consequence: **recouple rather than extend coupling time** — which cuts directly against extended coupling as the default lever for difficult positions | Gln/Asn present, with long activation | — |
| **Multiple addition (double insertion)** | Premature Fmoc removal *during* coupling — e.g. excess base with TBTU/DIPEA — inserts the same residue twice, giving a **+1-residue** by-product. The deletion/truncation vocabulary does not cover insertion; look for it explicitly | Needs process metadata | — |
| **Steric hindrance** | α,α-disubstituted residues (e.g. Aib) and N-methylated residues couple slowly; vendor data show crude purity roughly doubling when coupling time is extended at the hindered position | Non-standard residues; β-branched runs | `agg.beta_branched_run` |

### 1.6 Deletion vs truncation vs insertion — distinctions sources blur

- **Deletion**: one residue missing, chain continued. Nearly identical to the target and hard
  to separate. In Fmoc SPPS the dominant cause is **incomplete Fmoc removal**, not failed
  coupling — see the note at the head of Axis 2.
- **Truncation**: chain stopped. Easier to separate, and **capping** deliberately converts
  deletions into truncations for this reason.
- **Insertion (+1)**: the same residue added twice, from premature Fmoc removal during
  coupling. Frequently omitted from the vocabulary entirely.
- **Segment-level deletion**: in convergent synthesis, a whole missing segment — *easier* to
  purify than a single-residue deletion.

A source reporting "deletion sequences" without saying whether capping was used is describing
a different process from one that caps. Record it.

### 1.7 Irreversible chain termination — a family, not one reaction

These share an outcome — a chain that **can grow no further** and survives to the end of the
synthesis — but have unrelated causes, so no single check covers them.

| Route | Cause | Sequence-detectable? |
|---|---|---|
| **Guanidination** | Excess uronium/aminium reagent (e.g. HATU) relative to amino acid guanidinates the free N-terminal amine. Fixed by stoichiometry (reagent slightly sub-stoichiometric) or pre-activation — though pre-activation raises racemization | No — a stoichiometry failure |
| **Pyroglutamate** | N-terminal Gln cyclises and stops chain growth | **Yes** |
| **Trifluoroacetylation** | Residual TFA activated by diimide acetylates the N-terminus, giving a stable trifluoroacetylated chain. Quaternary ammonium groups on chloromethyl resins bind TFA strongly and make this worse | No — a reagent/resin-history failure |
| **Diketopiperazine** | Not a cap but a loss: the chain is cleaved off the resin entirely | **Yes** (C-terminal dipeptide) |

> **This family is why the standing advice "a failure label usually means low crude purity,
> not cannot-be-made" must be applied conditionally.** For an outcome measured as crude
> purity, that advice holds. But DKP cleaves the peptide from the resin, and pyroglutamate
> and trifluoroacetylation cap it permanently — those are chain-destroying, not
> purity-reducing. **Condition the wording on the outcome measure rather than applying it as
> a blanket rewrite.**

---

## Axis 2 — Process and environment factors

These are what a source must disclose for its sequence-level claims to be interpretable. When
reviewing, **record which of these the source states and which it omits.**

> **Which step actually loses material — and why the field's vocabulary points at the wrong
> one.** The language of the field is coupling-centric: "coupling efficiency", "difficult
> *coupling*", per-coupling models. The reference literature says otherwise for Fmoc SPPS:
> modern coupling reagents are efficient enough that acylation failure is "generally limited
> to cases where the amine to be acylated is extremely hindered", and the main cause of
> deletion sequences is **incomplete Fmoc removal** — aggregation hinders diffusion to and
> from the reactive sites in both directions. Separately, the one large interlaboratory
> study in this literature found that **side reactions during post-assembly (cleavage and
> deprotection) were more significant than assembly itself**.
>
> Three practical consequences: the process variables that matter most are the
> **deprotection regime, base, time and temperature**, not reagent potency; a model built
> only on assembly-step features may be modelling the wrong stage; and changing the support
> does not fix aggregation, because "the root of the problem lies in the interactions
> between peptide chains".

| Factor | Why it matters | What to extract |
|---|---|---|
| **Resin and linker** | Determines C-terminal functionality (acid vs amide), DKP susceptibility, and swelling. CTC resists DKP; Wang, Rink amide behave differently | Resin type, linker, substitution |
| **Resin loading** | Lower loading increases inter-chain spacing and reduces aggregation. A frequently decisive variable | mmol/g |
| **Coupling reagent** | Carbodiimide (DIC + HOBt/Oxyma), aminium/uronium (HBTU, HATU, HCTU, COMU), phosphonium (PyBOP, PyOxim) differ in efficiency, racemization suppression and guanidination risk | Reagent, equivalents |
| **Base** | DIPEA, NMM, 2,4,6-collidine — directly modulates racemization and aspartimide | Identity, equivalents |
| **Solvent** | DMF, NMP, DCM, DMSO and mixtures. Affects solvation and aspartimide rate. Secondary-amine impurities in aged DMF cause premature deprotection — a documented cause of irreproducibility | Solvent, grade, additives |
| **Temperature / microwave** | Speeds coupling and can overcome aggregation, but increases racemization, aspartimide and Arg lactam | Temperature, heating method |
| **Coupling time, excess, double coupling** | Usually described as *the* lever for difficult positions, and vendor data do show large crude-purity gains at hindered residues. But **two findings cut against it**: prolonged activation dehydrates Gln and Asn (recouple instead of extending), and one 33-laboratory study found **no correlation between coupling time and percent correct product**. Treat "we extended the coupling" as a claim to check, not a fix to assume | Time, equivalents, single/double |
| **Resin swelling** | Measured, quantitative, and underused. Poor swelling is "a cause or an indication of poor couplings". Note the counter-intuitive direction: **DMF swelling falls as temperature rises** for several resins while DMSO/toluene rises — so DMF is a poor choice for hot SPPS for a measurable reason | mL/100 mg, at the working temperature |
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
| **Resin colour tests** (Kaiser/ninhydrin, chloranil) | Free amine present after coupling | **Gives false negatives under aggregation or steric hindrance** — i.e. blind in exactly the difficult sequences the difficult-sequence protocols are for. Inapplicable to N-terminal secondary amines (Pro → isatin/chloranil). N-terminal Asn/Asp/Ser give a brown colour that confounds the read. Bead colour read by eye |
| **Quantitative Fmoc release** | Per-cycle deprotection efficiency | The right measurement in principle, and the reference literature records that it was **attempted and abandoned** for insufficient accuracy. So the multiplicative-yield argument that motivates the whole field is, in that source, unmeasured |
| **Dibenzofulvene release peak shape** | Qualitative read of deprotection kinetics in continuous-flow UV | A sharp peak means rapid release; "a broad peak with extending tail is associated with the onset of aggregation problems within the matrix". A free, live, *qualitative* diagnostic — distinct from quantitative Fmoc release and much more available |
| **Resin shrinkage, observed** | Visible contraction of the peptide-resin during synthesis | A zero-cost live indicator of chain association. Explicitly reported alongside β-structure formation and NMR line-broadening |
| **AAA / Edman sequencing** | Composition or sequence confirmation | **Its own chemistry cleaves residual protecting groups**, which can make a poor synthesis appear correct. A confirmation method that partially repairs the sample it is confirming |

> **Three blind spots worth stating explicitly**, because each puts an unmeasured floor
> under the error rate of any dataset built on these methods:
>
> 1. **Mass and retention are not injective over sequence permutations.** In a blinded
>    interlaboratory test, a deliberately **reversed peptide of identical mass** was passed as
>    correct by 7 of 20 laboratories; the ones that caught it used Edman sequencing. The same
>    study spiked 30% oxidised Met into another peptide and only 7 of 20 detected it. So a
>    label of "observed mass matches ordered peptide" contains a real fraction of
>    **mislabelled successes** — the target can be the wrong molecule at the right mass.
> 2. **Aspartimide and β-aspartyl by-products co-elute** with the target, so a
>    chromatographic purity measure under-reports them.
> 3. **Deamidation is invisible** to MALDI-TOF and to quadrupole ESI; detecting it needed
>    ion-trap ESI.

**Thresholding.** Many datasets binarise a continuous purity measure at an arbitrary cut
(50% is common). Two near-identical peptides either side of the cut land in opposite classes.
Always record the threshold and whether the continuous value is recoverable.

**The practical consequence.** A liability is only learnable from a dataset if the assay can
see it. When a well-established chemical liability shows no statistical signal, the ordered
explanations are: (1) the assay cannot resolve it; (2) the protocol already suppressed it;
(3) it is genuinely rare in that sequence space. Only after excluding those is "the chemistry
is wrong" on the table — and it is rarely the answer.

---

---

## Axis 4 — Provenance and operator

Added because the largest effect reported in this literature fits nowhere on the first three
axes, and leaving it out of the taxonomy means leaving it out of reviews.

The body that ran eight annual interlaboratory round-robins concluded that **peptide quality
depends as much or more on the experience and skill of the personnel than on the instrument
or method used**. Their distributional data are the strongest empirical material in the
reference literature:

- One year: 17% of submitted samples contained **none** of the correct peptide; 28% of crude
  and 68% of purified samples exceeded 75% correct peptide.
- Another: 48% of crude and 21% of purified samples fell below 75% of the desired product.
- A third: racemization ran 2–8% across submissions, but **only peptides from one laboratory
  carried the high values**.

**Why this is a review obligation, not a taxonomy quibble.** That last point is a cohort
effect with a named stratum. A synthesizability dataset pooled from several facilities carries
a **laboratory effect large enough to dominate any sequence feature**, and any sequence
property correlated with facility will absorb it. "Source library" in a cohort checklist is
not the same thing as "laboratory".

**What to extract:** how many labs or operators contributed; whether characterisation was
central or per-site; whether the dataset records provenance at all. **Not detectable from
sequence, and not from process metadata as usually recorded** — which is precisely why it has
to be asked about explicitly.

---

## Convergent / segment condensation

A different constraint set from linear assembly, and the catalogue above does not cover it.

- The **C-terminal residue of an activated segment is classically limited to Gly or Pro**, to
  avoid epimerising the junction. This is a sequence-intrinsic constraint on *where you may
  cut*, not on whether the peptide can be made.
- Segment-level deletions differ from the target by a **whole segment**, so they are easier to
  purify than single-residue deletions — a genuine advantage of the approach.
- Reported loadings are far lower than for linear SPPS (~50 µmol/g for segment condensation).

When a source reports a convergent synthesis, its difficulty factors are about junction
choice and segment solubility, and a linear-assembly rule set does not apply to it.

---

## Reporting template

```markdown
## Source
<citation>  type: primary|vendor|textbook|multi-author reference
For a multi-author work, one row per chapter that reports numbers:
| chapter | authors | outcome measure | threshold | evidence grade |

## Factors claimed
| factor | axis | direction | tradeoff | mechanism | evidence | scope | detectable |

## Process disclosed / omitted
<resin, loading, linker, reagent, base, solvent, temperature, deprotection regime,
 capping, cleavage cocktail, platform>
<and: any statistical treatment? replicates? per-cycle data?>

## Provenance
<how many labs/operators; central or per-site characterisation; is provenance recorded?>

## Assessment
- supported (grade A/B):
- asserted without support (grade D):
- grade-E risks if carried forward:
- contradicted — between sources OR between sections of one source:
- reported nulls:
- what the outcome measure cannot see:
```

**Field notes.** `direction` is **per axis** — a residue can help chain assembly and harm
chiral purity at the same position, and one direction field cannot say that; use `tradeoff`
to name the countervailing finding. `evidence` grades the **chapter**, never the volume.
Add a `reconstruction` note on any quotation assembled from OCR fragments.
