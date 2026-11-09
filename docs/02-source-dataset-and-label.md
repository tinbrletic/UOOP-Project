# Source dataset, label definition, and the performance ceiling

Everything here was extracted directly from `ao2c02425_si_002.xlsx` (supporting information
of Gutman et al. 2022, *ACS Omega*, "Predicting the Success of Fmoc-Based Peptide Synthesis",
PDF in repo root) and cross-checked against `peptide_baza.csv`.

---

## 1. How the label is defined — read this before interpreting any model

Quoting the header of Supplemental Table S1 verbatim:

> We calculated the 'Abundance Ratio' as the sum of the relative abundance values of the m/z
> peaks matching the ordered peptide divided by the sum of the relative abundance values of
> all m/z peaks observed in the MS1 spectrum. We considered MS1 spectra with an Abundance
> Ratio **< 50%** as indicative of problematic (or failed) syntheses (**label = 0**) and MS1
> spectra with an Abundance Ratio **> 50%** as successful (**label = 1**).

Three consequences that constrain what this project can honestly claim:

**(a) The target is crude-mixture MS1 purity, not "can this peptide be made."**
A peptide labelled 0 was still synthesised — it just came off the resin impure. The
colloquial framing "non-synthesisable peptide" that appears in `smote_balance.py` output
strings is wrong and should be corrected. The right phrasing is *"problematic synthesis"* or
*"low crude purity."*

**(b) The threshold is arbitrary and binarises a continuous quantity.**
A peptide at 49% abundance ratio and one at 51% are placed in opposite classes despite being
nearly identical. This throws away information and puts an irreducible noise floor on any
classifier. If the continuous abundance ratio is ever recoverable, **regression on it will be
better posed than classification**, and the metric should be rank correlation, not accuracy.

**(c) The label carries a sequence-dependent measurement confound.**
MS1 relative abundance is not a quantitative measure of molar purity. Ionisation efficiency
in positive-mode ESI varies strongly with sequence — basic residues (H, K, R) carry charge
readily and ionise well; acidic and highly hydrophobic sequences do not. So sequence
composition influences the *measurement* as well as the *chemistry*.

> **This means a model that appears to learn "basic residues → successful synthesis" may
> partly be learning "basic residues → stronger MS signal → higher abundance ratio → label 1."**
> Any causal or mechanistic interpretation of charge/basicity features must carry this caveat.
> This is not a hypothetical concern: it is why the DLOmix line of work exists at all.

---

## 2. Dataset shape (as loaded in this repo)

| Property | Value |
|---|---|
| Rows (`peptide_baza.csv`) | 1771 |
| Label 1 (successful) | 1539 (86.9%) |
| Label 0 (problematic) | 232 (13.1%) |
| Sequence lengths | 8 – 25 |
| Modal length | 15 (1036 peptides, 58.5%) |

Length distribution is highly non-uniform:

```
len  8:   11      len 15: 1036
len  9:  157      len 16:   12
len 10:   57      len 17:    3
len 11:   44      len 18:    6
len 12:   94      len 19:   12
len 13:  137      len 20:   48
len 14:  144      len 21:    5
                  len 22:    3
                  len 24:    1   len 25: 1
```

**Implication:** lengths above 16 have 80 peptides total across nine length values. Any
length-dependent claim beyond ~16 residues rests on a handful of examples. And since
therapeutic peptides are routinely 30–40 residues, **this dataset does not support
extrapolation to real drug-length peptides.** State that limitation explicitly in the report.

---

## 3. The published performance ceiling — AUC ≈ 0.77

Supplemental Table S2 (sheet `S3` of the xlsx) is the paper's own benchmark of every single
feature and every pairwise combination. Top results:

| AUC ± SD | Features |
|---|---|
| 0.773 ± 0.015 | length, Janin |
| 0.770 ± 0.015 | length, Kyte-Doolittle |
| 0.756 ± 0.014 | length, b (basic count) |
| 0.755 ± 0.015 | length, Rose |
| 0.752 ± 0.013 | length, Engelman GES |
| 0.749 ± 0.016 | all, Rose |
| 0.748 ± 0.018 | b, Rose |
| 0.747 ± 0.018 | all, Janin |
| 0.742 ± 0.016 | s, Rose |
| 0.739 ± 0.013 | length, Eisenberg |

Best single features: **peptide length 0.668 ± 0.007**, **h (aliphatic-hydrophobic count)
0.655 ± 0.019**, **Rose 0.734 ± 0.017**, **Janin 0.725 ± 0.020**.

Features that are **no better than chance**:

| AUC ± SD | Feature |
|---|---|
| 0.482 ± 0.027 | right drop |
| 0.487 ± 0.015 | r — aromatic count (F, W, Y) |
| 0.496 ± 0.014 | p, right drop |
| 0.497 ± 0.017 | p — polar count (C, S, T) |
| 0.500 ± 0.014 | p, r |
| 0.505 ± 0.028 | a — acidic count (D, E) |

### The three things this table settles

1. **This is a weak-signal problem.** The ceiling with the authors' own best feature pair is
   ~0.77 AUC. Our Random Forest reaching 0.721 AUC is **not underperformance — it is roughly
   where the problem's ceiling is.** Conversely, any pipeline here reporting AUC > 0.85
   should be assumed leaky until proven otherwise.

2. **Every top-10 entry is length or a hydrophobicity scale** (Rose, Janin, Kyte-Doolittle,
   Engelman, Eisenberg). Adding more descriptor families has not been shown to help.
   **But see §5 — the length component does not survive stratification.**

3. **Aromatic, polar and acidic composition counts are worthless on this task** (AUC 0.48–0.51).
   Our dataset devotes nine columns to compositional counts — most of which the source paper
   already demonstrated carry no signal, *and* which are mislabelled (see audit Finding 1).

---

## 4. The paper's residue groupings differ from ours

Legend from Supplemental Table S2, verbatim:

| Symbol | Definition in the paper |
|---|---|
| `s` / `S` | small residues — **A, G, P** |
| `h` | aliphatic-hydrophobic — **I, L, M, V** |
| `h stretch` | **highest number of consecutive contiguous aliphatic-hydrophobic residues** |
| `a` | acidic — **D, E** |
| `l` | large — **N, Q** |
| `r` | aromatic — **F, W, Y** |
| `p` | polar — **C, S, T** |
| `b` | basic — **H, K, R** |
| `left drop` | loss of a single residue, typically at the N-terminus |
| `all drop` | residue loss at any position |
| `right drop` | residue loss at the C-terminus |
| `all bond` | the bond to the right of a dropped residue; 20 × 20 = 400 possible bonds |

Our `scrpit.py` uses *standard textbook biochemical groupings* instead (e.g. Small =
A,C,D,G,N,P,S,T,V; Aliphatic = A,I,L,M,V; Aromatic = F,W,Y,H). **These are different sets.**
Neither is "wrong" as chemistry, but they are not comparable, so our compositional numbers
cannot be read against the paper's AUC table. Pick one convention and document it.

Note also `all bond` — 400 features for amino-acid bonds. That is the same 20×20 construction
as our `X8_*` dipeptide features in `add_features.py`.

### The feature we never implemented

**`h stretch` — the longest consecutive run of I/L/M/V — is in the paper's feature set and
absent from ours.** It is also the only feature in the list with a direct mechanistic story:
consecutive hydrophobic residues promote inter-chain association on the resin, which hinders
reagent access at the growing chain. Run-length is exactly the information that mean
hydrophobicity and total counts throw away.

**Adding stretch/run-length features is the highest-value, lowest-effort improvement available
to this project.**

---

## 5. The length-15 cohort confound — and what survives it

This was found by stratifying the dataset. It is the most consequential result in these docs.

### Failure rate is not monotone in length

| Length band | n | failures | failure rate |
|---|---|---|---|
| 8–14 | 644 | 23 | **3.6%** |
| **15** | **1036** | **202** | **19.5%** |
| 16–19 | 33 | 3 | 9.1% |
| 20–25 | 58 | 4 | 6.9% |

**Length-15 peptides are 58.5% of the dataset and carry 202 of 232 failures — 87% of them.**
Their failure rate is 5.4× that of the 8–14 band. The 20–25 band, which a chain-length effect
would predict to be the *worst*, is close to the best.

This is the signature of a **cohort effect**: the length-15 peptides are evidently a distinct
sub-library — a different campaign, instrument, scale or operator — and whatever differs about
them dominates the label. Length then acts as a near-perfect indicator variable for that
cohort, not as a measure of chain length.

### What this does to the paper's headline feature

Gutman et al. report **peptide length as the strongest single feature, AUC 0.668 ± 0.007**,
and their best pair as **length + Janin, AUC 0.773 ± 0.015**.

Stratified results on the same data (AUC for predicting *failure*):

| Stratum | n | failures | length | **Janin mean** | KD mean | longest I/L/M/V run |
|---|---|---|---|---|---|---|
| All | 1771 | 232 | 0.639 | 0.734 | 0.717 | 0.606 |
| **Length 15 only** | 1036 | 202 | **0.500** | **0.740** | 0.727 | 0.581 |
| Length ≠ 15 | 735 | 30 | 0.599 | 0.639 | 0.591 | 0.569 |

**Conclusion, stated carefully:**

- **Hydrophobicity is real signal.** Janin mean reaches AUC **0.740 within the homogeneous
  length-15 cohort** — marginally *better* than on the full dataset. The aggregation mechanism
  holds within a cohort, where no batch indicator is available to the model. This independently
  reproduces the paper's reported Janin AUC of 0.725.
- **Length's apparent power is cross-cohort.** It is 0.500 within length-15 by construction,
  and 0.599 in the remainder (on only 30 failures, so a wide interval). The reported 0.668 is
  therefore substantially a cohort indicator, and **"length + Janin = 0.773" is best read as
  Janin doing the work plus a batch label.**
- **Do not include raw length as a feature on this dataset** without reporting the stratified
  result alongside it. A model that appears to beat the published AUC by exploiting length is
  not predicting chemistry.

### What to do about it

1. Report every result **stratified by length**, or at minimum report the length-15 result
   separately, since that stratum contains the overwhelming majority of the signal.
2. Ask the source authors — or infer from the supporting information — whether the length-15
   set is a separate synthesis campaign. If it is, it belongs in the methods section as a
   confound.
3. Treat the length-15 stratum as the primary analysis set: n=1036 with 202 failures (19.5%)
   is both larger and far better balanced than the full set at 13.1%.
4. **The honest headline for this project** is: mean hydrophobicity predicts crude-purity
   failure at AUC ≈ 0.74 within a homogeneous cohort, consistent with the β-sheet aggregation
   mechanism documented in Houben-Weyl E22a §1.2.1.3. That is a defensible, mechanistically
   grounded claim. It is a better result than a higher number obtained by leaking a batch
   indicator.
