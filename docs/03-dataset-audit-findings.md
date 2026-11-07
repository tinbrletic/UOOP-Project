# Dataset and pipeline audit — verified findings

Audited against `develop` (commit `2c4b470`, restructured layout). Every finding was
confirmed by re-running the repo's own feature computation and comparing against the
committed CSVs — not inferred from reading code.

**Status key:** `OPEN` = present in current code. `FIXED` = resolved in the main pipeline.
`LEGACY` = resolved in `src/feature_selection.py`, still present in `experiments/` or
legacy scripts.

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Compositional feature columns mislabelled — 8 of 9 hold the wrong quantity | **Critical** | **FIXED** in `build_dataset.py` |
| 2 | Credentials committed in plaintext | **High** | **PARTIAL — rotate the password** |
| 3 | No run-length / positional features despite source paper using them | **High** | **FIXED** in `sequence_features.py` |
| 4 | Generation script does not reproduce the committed CSV | Medium | **FIXED** (reason corrected — see below) |
| 5 | Near-duplicate sequences not checked before splitting | Medium | **OPEN** |
| 6 | `peptide_baza_full.csv` has a corrupted, double-quoted header | Low | **OPEN** |
| 7 | SMOTE applied before splitting, persisted to disk | Critical | **LEGACY** |
| 8 | `scoring='r2'` used to evaluate classifiers | High | **LEGACY** |
| 9 | Feature selection fitted on full dataset before split | High | **LEGACY** |
| 10 | Headline metrics dominated by majority class | High | **FIXED** |

---

## Finding 1 — Compositional columns are mislabelled *(Critical, FIXED)*

> **Fixed** by `src/build_dataset.py`, which names every column for what it contains
> (`n_paper_h`, `n_biochem_small`, …). The mis-mapping below is reproduced deliberately by
> `--legacy-check` as a regression test. The historical CSVs under `data/` still carry the
> fault and should not be used.

`src/ingest_excel_to_db.py`

### The bug

`compositional()` appends **two** values per group — a count *and* a fraction:

```python
for group in groups:
    count = sum(seq.count(amino) for amino in group)
    properties.append(count)             # value 2i
    properties.append(count / len(seq))  # value 2i+1
```

With 9 groups it returns **18** values. The `INSERT` at line 95 then writes
`comp[0] … comp[8]` — the **first 9 of 18** — into nine differently-named columns.

### Verification

Replicating the computation for the first three sequences reproduces the committed
`data/raw/peptide_baza.csv` columns exactly:

```
SELLTPLGIDLDEW
  compositional() returns : [3, 0.2143, 6, 0.4286, 5, 0.3571, 1, 0.0714, 8, 0.5714, 6, ...]
  comp[0:9] written to DB : [3, 0.2143, 6, 0.4286, 5, 0.3571, 1, 0.0714, 8]
  CSV row                 :  3, 0.2143, 6, 0.4286, 5, 0.3571, 1, 0.0714, 8   ← exact match
```

Confirmed identically for `KEITVATSRTLSYY` and `KENSYTTTIKPVTY`.

### What each column actually contains

| Column name | Name implies | **Actually contains** |
|---|---|---|
| `tiny_group` | count of Tiny | count of Tiny (A,C,G,S,T) — **correct** |
| `small_group` | count of Small | **fraction** of Tiny |
| `aliphatic_group` | count of Aliphatic | **count of Small** (A,C,D,G,N,P,S,T,V) |
| `aromatic_group` | count of Aromatic | **fraction of Small** |
| `non-polar_group` | count of Non-polar | **count of Aliphatic** (A,I,L,M,V) |
| `polar_group` | count of Polar | **fraction of Aliphatic** |
| `charged_group` | count of Charged | **count of Aromatic** (F,W,Y,H) |
| `basic_group` | count of Basic | **fraction of Aromatic** |
| `acidic_group` | count of Acidic | **count of Non-polar** (A,C,F,G,I,L,M,P,V,W,Y) |

**Polar, Charged, Basic and Acidic counts are absent from the dataset entirely.**

### Why it matters beyond the names

- **Every feature-importance interpretation in `results/` is wrong.** A reported top feature
  of `non-polar_group` is really *the count of small residues*. Any sentence in the report
  naming these features states a false result.
- **Four of nine columns are fractions of a count that is also a column**, and `peptide_len`
  is a column too — near-deterministic collinearity fed straight into feature selection,
  destabilising RFECV and SelectFromModel rankings.
- It propagates to every derived file: `data/processed/peptide_baza_formatted.csv`,
  `_full.csv`, `_balanced.csv`, and all of `results/`.

### Fix

Choose one grouping convention — recommend the **source paper's** (see
[02](02-source-dataset-and-label.md) §4) so results are directly comparable to its published
AUC table — emit counts and fractions under distinct explicit names (`n_small`,
`frac_small`, …), and **regenerate from `data/raw/ao2c02425_si_002.xlsx`.** Do not patch the
CSVs by renaming columns.

Also fix the stale comment `#3 Aromatic (removed H, per Wikipedia)` — `H` is still in that tuple.

---

## Finding 2 — Credentials committed in plaintext *(High, PARTIAL)*

> **Partly addressed.** `src/ingest_excel_to_db.py` is now a deprecation stub containing no
> credential, and `src/build_dataset.py` needs no database at all, so nothing in the
> pipeline requires the password any more. **Step 1 below is still outstanding and still
> matters:** the value remains in git history across several merged pull requests.

`src/ingest_excel_to_db.py:65` contains a hardcoded MySQL password in version control.

**Actions, in order:**
1. **Rotate the password now.** Assume it is compromised — it is in git history on a
   repository with multiple branches.
2. Replace with `os.environ["PEPTIDE_DB_PASSWORD"]` and document the variable in the README.
3. Note that deleting it in a new commit does **not** remove it from history. If this
   repository will ever be public, the history needs rewriting (`git filter-repo`).

---

## Finding 3 — No positional or run-length features *(High, FIXED)*

> **Fixed** in `src/sequence_features.py`. Measured on the rebuilt dataset, within the
> length-15 stratum: `homopolymer_run` AUC 0.606 and `max_pbeta_window6` 0.599, both of
> which beat the source paper's own `h stretch` (0.581). `peptide_len` is 0.500 in the same
> stratum, confirming it carries no within-cohort information.

Every descriptor is a whole-sequence aggregate: a mean hydrophobicity, a total count, a
one-hot of the first residue, or a dipeptide tally. None encodes *where* residues sit or how
they cluster.

`ILVMAAAAAA` and `AAILAAVMAA` receive **identical** values for every mean-based and
count-based descriptor, despite behaving very differently during chain assembly.

This is not a theoretical gap. Houben-Weyl E22a §1.2.1.3 ("Difficult Sequences") identifies
sequence-dependent **intermolecular association into β-sheet-like structures**, which renders
the chain termini inaccessible to further reaction, as the dominant cause of incomplete
coupling. Run length is precisely the quantity that captures this and that means and totals
discard.

The source paper includes exactly such a feature — **`h stretch`, the longest consecutive run
of I/L/M/V** — and this project never implemented it.

Cheap additions, highest expected value in the project:

- longest run of aliphatic-hydrophobic residues (I, L, M, V) — the paper's `h stretch`
- longest run of β-branched residues (I, V, T); longest homopolymer run
- windowed maximum hydrophobicity (widest 5-residue window mean), not only the global mean
- `X8_DG` and the other aspartimide-relevant dipeptides treated as named features rather
  than 2 of 400 anonymous columns
- position-from-C-terminus weighting — later residues sit on a longer, more crowded chain

See `.claude/skills/spps-synthesizability-review/` for an implementation of these.

---

## Finding 4 — Generation script does not reproduce the committed data *(Medium, FIXED)*

> **Correction.** An earlier version of this finding attributed the mismatch to the Cornette
> hydrophobicity column, claiming it held 1598 distinct values while the script wrote `None`.
> **That was wrong** — it came from an off-by-one in a column index: field 27 is
> `hydrophobic_cornette` and holds exactly **one** distinct value (`0.0`), while field 28 is
> `hydrophobic_eisenberg`, which is the column with 1598 values. The conclusion happened to
> be right for an entirely different reason, found below. The error is left on the record
> rather than quietly rewritten.

### The actual cause: the Aliphatic group changed and the data was never regenerated

`data/raw/peptide_baza.csv` was generated with `Aliphatic = (A, I, L, V)`. Commit `388d6b4`
("Correct compositional property categorization…") changed it to `(A, I, L, M, V)`, adding
methionine, and the CSV was never rebuilt. The committed script and the committed data
therefore disagree on the **468 rows containing M**, in exactly the two columns fed by that
group.

Verified by exhaustive reconstruction: rebuilding every historical value from
`data/raw/ao2c02425_si_002.xlsx` under the hypothesis *"compositional mis-mapping (Finding 1)
plus methionine-free Aliphatic"* reproduces **46,046 of 46,046 cells with zero mismatches**.
Under the committed script's own grouping it mismatches 936 cells, all in `non-polar_group`
and `polar_group`.

### `hydrophobic_cornette` is a dead column

It is constant `0.0` in all 1771 rows. The `peptides` library at the pinned version
**0.5.0 has no Cornette scale at all** — `p.hydrophobicity("Cornette")` raises
`Invalid hydrophobicity scale`. That is why the original author commented it out and
substituted `None`. A constant column is not a feature, and this one cannot be computed with
the pinned dependency, so it is dropped rather than repaired.

### Fix

`src/build_dataset.py` replaces the MySQL path entirely. Its `--legacy-check` mode performs
the reconstruction above as a regression test, so the claim is re-verifiable:

```bash
python src/build_dataset.py --legacy-check
```

---

## Finding 5 — Near-duplicate sequences not checked *(Medium, OPEN)*

No deduplication or similarity check runs before splitting. If the source library contains
families of related sequences, near-identical peptides can straddle the train/test boundary
and inflate scores even with correct stratified CV. Check exact duplicates first, then
cluster by sequence identity and **split at cluster level**.

---

## Finding 6 — `data/processed/peptide_baza_full.csv` header is corrupted *(Low, OPEN)*

Its header is double-quoted so the whole row parses as one field:

```
"id;""peptide_seq"";""peptide_len"";""synthesis_flag"";...
```

Regenerate or delete it.

---

## Findings 7–9 — Resolved in the main pipeline, still live in `experiments/` *(LEGACY)*

`src/feature_selection.py` now does this correctly, and that deserves acknowledgement:

- `RepeatedStratifiedKFold` for splitting
- SMOTE applied **per-fold inside an `imblearn` `Pipeline`**, training data only
  (comment at lines 37–39 states this explicitly)
- `roc_auc_score` and `matthews_corrcoef` among reported metrics
- feature selection integrated into the pipeline (`SelectFromModel`, `RFECV` as pipeline steps)

Still outstanding:

- **`src/smote_balance.py`** persists a pre-balanced CSV. Its synthetic rows are unusable
  for a second reason beyond leakage: the sequence strings are **randomly generated and
  unrelated to their own feature vectors** (`random.choices(amino_acids, …)` overwrites
  `peptide_seq` after SMOTE produced the features). SMOTE also interpolates the one-hot
  `X4_*` and count `X5_*`/`X8_*` columns into fractional values that correspond to no
  peptide. Mark the script deprecated or delete it, and do not cite any result derived from
  `peptide_baza_balanced.csv`.
- **`experiments/10_fold_cross_validation.py`** still uses `scoring='r2'` on classifiers.
- **`experiments/fs_seq.py`** still fits the selector on the full dataset before splitting.
- `results/archive/` contains outputs from both eras. Label clearly which runs are
  superseded so no one cites them.

---

## Finding 10 — Majority-class metrics *(FIXED — keep it that way)*

The historical `results.txt` reported:

```
SVM            Accuracy 0.869   F1 0.930   Recall 1.000   MCC 0.000
Random Forest  Accuracy 0.868   F1 0.928   Recall 0.979   MCC 0.205
               Confusion matrix:  [[  30  202]
                                   [  32 1507]]
```

The SVM was a **constant classifier** — recall 1.000 with MCC 0.000 means it predicted
"successful" for every peptide, and its accuracy of 0.869 is exactly the base rate
(1539/1771). Its F1 of 0.930 was the best number on the page.

The current pipeline reports MCC and AUC-ROC, which exposes this. Two things to hold:

1. **Keep accuracy and positive-class F1 out of the headline.** They are vanity metrics at
   13% minority.
2. **Report with the failure class as positive**, and add **average precision (PR-AUC)**.
   A chemist asking "which of my peptides will give me trouble?" wants failure-class recall.
   In the run above that number was 30/232 = **12.9%**.

---

## Recommended order of work

1. **Finding 2** — rotate the password. Minutes, and the clock is running.
2. **Finding 1** — regenerate the dataset with correct, explicitly named features.
   Everything downstream depends on it.
3. **Finding 3** — add run-length and windowed features. Most likely source of new signal.
4. **Findings 7–9** — deprecate the legacy scripts so they cannot be rerun or cited.
5. **Findings 4, 5, 6** — reproducibility and hygiene.

Baseline to beat, from the source paper: **AUC 0.773** using length + Janin hydrophobicity
alone. Any model that does not clear a two-feature logistic regression is not earning its
complexity.
