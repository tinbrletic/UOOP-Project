# Demo — five minutes, three modes

Every command below runs from the skill directory with no setup beyond Python 3.8+.
The outputs shown are **real output, trimmed for length** — not illustrations.

```bash
cd spps-synthesizability-review
```

---

## 1. Self-test — does the rule set behave?

Run this first on any new install, and whenever you change a rule.

```bash
python3 scripts/scan_peptides.py --selftest
```

It scans sequences the literature names as known-difficult, plus benign controls.
**A difficult case that draws nothing, or a control that lights up, is a rule defect.**

```
AAAAAAAAAA                       CRITICAL  decaalanine - the canonical difficult sequence
    [CRITICAL] agg.homopolymer_run
    [MEDIUM  ] agg.hydrophobic_window
    [MEDIUM  ] agg.no_disruptor_window
...
AGSGAGSGAK                       INFO      benign control - Gly/Ser rich, should stay quiet
    (nothing)

EEKKEEKKEEKK                     LOW       benign control - charged, highly soluble
```

Decaalanine is the case worth watching. It contains no residue from any hydrophobic set, so a
rule set built on hydrophobic-residue identity scores it zero — and it is the founding
difficult sequence in the reference literature (HW-E22a p726). It fires here through
`agg.homopolymer_run`, which encodes **backbone regularity** instead.

---

## 2. Screen sequences

```bash
python3 scripts/scan_peptides.py demo/example.fasta
```

`demo/example.fasta` holds ten sequences chosen to exercise each rule family: two
literature-named difficult cases, a natural 30-mer, contrived aliphatic runs, three
side-reaction motifs, and two benign controls.

Findings are split into three groups that answer **different questions** — this separation is
the point, not decoration:

```
CHAIN-ASSEMBLY DIFFICULTY  (solvation / steric access)
  agg.hydrophobic_window          7  ( 70.0% of sequences)
  agg.anchor_zone                 5  ( 50.0% of sequences)
  agg.h_stretch                   4  ( 40.0% of sequences)
  agg.pseudoproline_site          2  ( 20.0% of sequences)
  agg.homopolymer_run             1  ( 10.0% of sequences)

CUMULATIVE YIELD  (length; distinct from aggregation)
  agg.length                      1  ( 10.0% of sequences)

PROCESS LIABILITIES  (discrete, each with a mitigation)
  side.diketopiperazine           7  ( 70.0% of sequences)
  side.aspartimide                3  ( 30.0% of sequences)
  side.pyroglutamate              3  ( 30.0% of sequences)
  side.racemization               3  ( 30.0% of sequences)
```

Per-sequence output names positions and mechanisms, worst first:

```
QLLLLVDGKP   (len 10, h_stretch 5, Janin -0.050)
  [CRITICAL] Aliphatic-hydrophobic run of 5 (I/L/M/V) at position 2. Strong on-resin
             aggregation risk; expect incomplete couplings through this region.
  [CRITICAL] Asp-G at position 7 (sensitivity +++++). Repeated piperidine exposure drives
             aspartimide formation and its by-products.
  [HIGH    ] Hydrophobic 5-residue window (mean Kyte-Doolittle 3.88) at position 2: LLLLV
  [HIGH    ] N-terminal Gln. Cyclises to pyroglutamate, capping the chain.
  [HIGH    ] Pro at C-terminal position 10. DKP formation cleaves the peptide from the resin
             during Fmoc removal; 2-chlorotrityl resin is the standard mitigation.
```

**Two things to notice, both honest limitations rather than selling points:**

- `side.diketopiperazine` fires on 7 of 10. It triggers on Gly or Pro anywhere in the
  C-terminal dipeptide, which is common. High prevalence means low specificity — treat it as
  a protocol note (use 2-chlorotrityl resin when Pro is C-terminal) rather than a risk score.
- `agg.anchor_zone` on a 10-mer reports a zone of **two residues**, because the window is
  measured 8–25 positions from the C-terminal anchor and a 10-mer barely has one. This is the
  degeneracy the catalogue warns about, visible in the output. The rule is near-useless on
  short peptides and says so when you calibrate.

Other formats: `--format json` for machine consumption, `--format csv` for a per-sequence
feature table. Input can be CSV, TSV, XLSX, FASTA or plain text; the sequence column is
auto-detected, or name it with `--column`.

---

## 3. Calibrate on your own labelled data

This is the mode that matters, and the reason the scanner ships **no hardcoded effect sizes**.

```bash
python3 scripts/scan_peptides.py mydata.csv --label-column outcome
```

Any table with a sequence column and an outcome column works. The worked example below uses
1771 peptides labelled by MS1 crude purity:

```
n = 1771   outcome column = 'target_col'   value counted as failure = '0'
failures = 232 (13.1%)   successes = 1539
```

### It measures its own severity tiers

Severity labels are assigned from mechanism, so whether they rank *risk on your data* is an
empirical question:

```
worst tier      n   fail  fail rate  vs base
CRITICAL       73     15      0.205    1.57x
HIGH          843    137      0.163    1.24x
MEDIUM        682     72      0.106    0.81x   <- at or below base rate
INFO          139      4      0.029    0.22x
(all)        1771    232      0.131
```

On this dataset **MEDIUM sits below the base rate** — it means "unremarkable", not "at risk".
That is why the default listing floor is HIGH, and why the tool prints this instead of quietly
flagging 90% of sequences.

### It runs a cohort check automatically

```
WARNING: length 15 holds 87% of all failures and 58% of all rows.
That is a distinct sub-library, not a chain-length effect.

Re-checking within length == 15 (n=1036):
  FEATURE            AUC all AUC in-stratum
  length              0.6391    0.5000
  janin_mean          0.7336    0.7398
  kd_mean             0.7169    0.7270
  max_kd_window       0.7152    0.6965
```

**This is the single most useful thing the tool does.** Length looks like the strongest
feature at AUC 0.639 and carries no information at all within a homogeneous cohort — it was
acting as an indicator for a sub-library. Mean hydrophobicity holds up, slightly higher
in-stratum. A feature whose AUC collapses was riding the cohort; one that holds is measuring
something real.

The check fires on two different structures: one value dominating both rows and failures, and
a monotone gradient across length bands where no single value dominates.

### It says when a rule could not have been tested

```
RULE                       fires fail|fire  fail|not    lift power
agg.hydrophobic_window       608     0.238     0.075    3.19
agg.homopolymer_run            3     0.333     0.131    2.55 UNTESTED (exp. 0.4 failures)
side.arg_lactam              923     0.098     0.167    0.58
```

Three readings, all needed:

- `agg.hydrophobic_window` at **lift 3.19** is a real effect on this data.
- `agg.homopolymer_run` shows lift 2.55 but fires on **3 of 1771 rows** — it expects 0.4
  failures, so the number is counting noise. **Untested, not refuted.** The tool will not let
  you read it as a result, and the correct response is not to retune the threshold.
- `side.arg_lactam` at **lift 0.58** fires *more* often on successes. Arg is basic, basic
  residues ionise well in positive-mode ESI, and this label is MS-derived — so the "effect"
  is plausibly in the detector. The tool flags the inversion; diagnosing it is yours.

---

## What to try next

| Goal | Command |
|---|---|
| Check the rules still behave after an edit | `--selftest` |
| Screen a candidate set before synthesis | `scan_peptides.py peptides.fasta` |
| Find out which rules work on *your* assay | `--label-column <col>` |
| Feed another tool | `--format json` |
| See every option | `--help` |

If you have no labelled data, the documented fallback is to **run the scanner on the failure
cases a source itself names** — free validity checking against ground truth the author hands
you. That is how two defects in this rule set were found; see `references/validation.md`.
