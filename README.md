# UOOP-Project — Predicting Fmoc Peptide Synthesis Success

Academic ML study (Izborni Projekt — Evolucijsko racunarstvo) reproducing and extending Gutman et al. 2022, *Predicting the Success of Fmoc-Based Peptide Synthesis*. The pipeline computes physicochemical, compositional, and sequence-composition features for peptides from the paper's supplementary data, balances the class distribution with SMOTE inside the cross-validation loop, then benchmarks six classifiers under multiple feature-selection strategies.

The original paper is at `docs/paper.pdf`. See `CLAUDE.md` for in-depth implementation notes (caching layer, statistical post-hocs, gotchas).

## Layout

```
src/                       Main pipeline scripts (run from project root)
  build_dataset.py         xlsx -> peptide_features.csv. No database. START HERE
  sequence_features.py     Compositional, positional and motif features (pure functions)
  ingest_excel_to_db.py    DEPRECATED stub -> build_dataset.py
  add_features.py          peptide_baza.csv -> peptide_baza_formatted.csv (+440 seq features)
  smote_balance.py         Pre-balanced CSV (legacy; main run now SMOTEs in-CV)
  feature_selection.py     Main experiment: 10x10 CV, FS sweeps, stats, ROC/AUC
  diagnostics/             smote_report.py, verify_balance.py (read-only checks)
experiments/               Earlier standalone work (fs_rand_for, fs_seq, 10-fold)
data/raw/                  Source xlsx + raw DB export CSV
data/processed/            Generated CSVs (peptide_features = current; others historical)
docs/                      Paper PDF, slides, molecule image
results/                   Run outputs
  smote_in_cv/<ts>/        Current-run artifacts (created by feature_selection.py)
  archive/                 Older runs, prior layouts
cache/                     Auto-generated (gitignored): parquet, CV splits, run pickles
```

## Quickstart

```powershell
# 1) Create + activate venv (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Run the main experiment (always from project root)
python src/feature_selection.py
```

Outputs land in `results/smote_in_cv/<timestamp>/`. To bypass the cache, set `NO_CACHE=1` or pass `--no-cache`.

## Pipeline (linear chain)

1. `src/build_dataset.py` — `data/raw/ao2c02425_si_002.xlsx` -> `data/processed/peptide_features.csv`. No database, no credential, one command.

   ```bash
   python src/build_dataset.py                 # build the feature table
   python src/build_dataset.py --legacy-check  # prove it reproduces the historical CSV
   ```

   `--legacy-check` rebuilds every value of `data/raw/peptide_baza.csv` from the spreadsheet and diffs: **46,046 cells, 0 mismatches**, reproducing both historical faults (the compositional mis-mapping, and the methionine-free Aliphatic group). It is the regression test for this script.

   `--convention paper` (default) uses Gutman et al.'s residue groupings so results compare directly to that paper's AUC table; `--convention biochem` uses standard textbook groupings. They are **not** interchangeable — pick one and say which.

   > The old `src/ingest_excel_to_db.py` is a deprecation stub. It routed through MySQL, which is why the repository carried a hardcoded password (**still in git history — rotate it**), and its compositional columns were mislabelled. See `docs/03-dataset-audit-findings.md`.

2. `src/add_features.py` — `data/raw/peptide_baza.csv` -> `data/processed/peptide_baza_formatted.csv` (adds 440 X4/X5/X8 features). Note this still reads the **historical** CSV with the mislabelled columns.
3. `src/smote_balance.py` — Produces `data/processed/peptide_baza_balanced.csv` (legacy; pre-balancing leaks across folds — `feature_selection.py` now SMOTEs inside CV instead).
4. `src/feature_selection.py` — Main run. Reads `data/processed/peptide_baza_formatted.csv` and writes everything to `results/smote_in_cv/<ts>/`.

## Notes

- All scripts use relative paths and assume CWD = project root. Don't run them with `cd src && python feature_selection.py` — paths to `data/` and `cache/` won't resolve.
- The target column is named `target_col` in MySQL-era scripts and `targetcol` everywhere downstream of `add_features.py` (the rename happens there).
- Platform: Windows PowerShell. Use `;` not `&&` to chain commands.
