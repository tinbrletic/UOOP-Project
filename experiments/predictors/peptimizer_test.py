"""Score peptides with peptimizer's pretrained fast-flow synthesis model.

Uses the 'minimal' predictor (learningmatter-mit/peptimizer), which needs only the
pre-synthesized chain + incoming amino acid - no machine parameters - and predicts
`first_diff` = deprotection UV peak height - width. Aggregation broadens/flattens
that peak, so MORE NEGATIVE first_diff = more aggregation = harder coupling.

Each peptide is scored per coupling step and summarised (min = worst step, mean).

Run from the project root with the dedicated Python 3.10 venv:

    experiments/predictors/.venv-peptimizer/Scripts/python.exe \
        experiments/predictors/peptimizer_test.py

Setup (peptimizer is not vendored - see .gitignore):

    git clone https://github.com/learningmatter-mit/peptimizer.git experiments/predictors/peptimizer
    py -3.10 -m venv experiments/predictors/.venv-peptimizer
    experiments/predictors/.venv-peptimizer/Scripts/python.exe -m pip install \
        "tensorflow==2.10.1" rdkit pandas "scikit-learn==1.1.3" h5py matplotlib "numpy==1.26.4"

Why this venv: peptimizer targets Python 3.7 / TF 2.0. The project's own 3.13 venv can
only get TF 2.20+ (Keras 3), which will not load the 2019-era .hdf5 model. TF 2.10 on
3.10 keeps Keras 2. numpy must stay <2 or the TF 2.10 binary fails to import.

Conventions verified against peptimizer's own training data:
  - `pre-chain` is written C-terminus-first (it grows by appending, and SPPS adds each
    residue to the N-terminus), so our standard N->C sequences are REVERSED here.
  - first_diff == first_height - first_width exactly.
  - Feature construction reuses peptimizer's own FeatureTransformation so the (buggy)
    Morgan radius=fp_bits call path matches how the model was trained. Do not "fix" it.
Sanity check: this pipeline reproduces peptimizer's own labels at r = 0.69 / R^2 = 0.48.
"""
import csv
import datetime
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PEPTIMIZER_DIR = os.path.join(PROJECT_ROOT, "experiments", "predictors", "peptimizer")
CSV_SRC = os.path.join(PROJECT_ROOT, "data", "raw", "peptide_baza.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "predictors")
SEQ_MAX = 50

if not os.path.isdir(PEPTIMIZER_DIR):
    raise SystemExit(f"peptimizer clone not found at {PEPTIMIZER_DIR} - see docstring")

# peptimizer's modules do app_dir = os.getcwd(), so run from inside the clone
os.chdir(PEPTIMIZER_DIR)

# shim: peptimizer's pickles were written by sklearn <0.22 (modules since renamed)
import sklearn.preprocessing._data  # noqa: E402

sys.modules["sklearn.preprocessing.data"] = sklearn.preprocessing._data
try:
    import sklearn.preprocessing._label  # noqa: E402

    sys.modules["sklearn.preprocessing.label"] = sklearn.preprocessing._label
except Exception:
    pass

sys.path.append(os.path.join(PEPTIMIZER_DIR, "utils", "utils_synthesis"))
sys.path.append(os.path.join(PEPTIMIZER_DIR, "utils", "utils_common"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from synthesis_feature_transformation import FeatureTransformation  # noqa: E402
from synthesis_predictor import Predictor  # noqa: E402


def read_dataset(path=CSV_SRC):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [(r["peptide_seq"].strip(), r["synthesis_flag"].strip().upper())
                for r in csv.DictReader(fh, delimiter=";")]
    return [(s, f) for s, f in rows if f in ("TRUE", "FALSE")]


def build_steps(seqs):
    """One row per coupling step: pre-chain (C->N) + incoming residue."""
    rows = []
    for idx, seq in enumerate(seqs):
        rev = seq[::-1]  # synthesis order: C-terminus first
        if len(rev) > SEQ_MAX:
            raise SystemExit(f"sequence longer than seq_max={SEQ_MAX}: {seq}")
        for k in range(1, len(rev)):
            rows.append({"peptide_index": idx, "sequence": seq, "step": k,
                         "pre-chain": rev[:k], "amino_acid": rev[k]})
    return pd.DataFrame(rows)


def score(seqs):
    steps = build_steps(seqs)
    ft = FeatureTransformation(
        model_type="minimal", mode="predict", fp_radius=3, fp_bits=128, seq_max=SEQ_MAX,
        pre_chain_smiles_path="dataset/data_synthesis/pre_chain_smiles.json",
        amino_acid_smiles_path="dataset/data_synthesis/amino_acid_smiles.json",
        transformation_functions_path="dataset/data_synthesis/transformation_function.pkl",
    )
    nnX, _ = ft.scale_transform(steps.copy())
    predictor = Predictor(
        model_type="minimal",
        model_path="model/model_synthesis/synthesis_minimal.hdf5",
        scaling_functions_path="dataset/data_synthesis/scaling_function.pkl",
    )
    steps["first_diff"] = predictor.predict(nnX).ravel()
    return steps


def summarise(steps, seqs):
    g = steps.groupby("peptide_index")["first_diff"]
    out = []
    for idx, seq in enumerate(seqs):
        v = g.get_group(idx)
        out.append({"sequence": seq, "length": len(seq), "n_steps": len(v),
                    "min_first_diff": round(float(v.min()), 4),
                    "mean_first_diff": round(float(v.mean()), 4),
                    "n_steps_below_-0.4": int((v < -0.4).sum())})
    return out


def auc_roc(scores, labels):
    """AUC via rank identity, tie-corrected. labels: 1 = positive."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    data = read_dataset()
    false_seqs = [s for s, f in data if f == "FALSE"]
    true_seqs = [s for s, f in data if f == "TRUE"]

    print(f"FALSE peptides: {len(false_seqs)}")
    f_steps = score(false_seqs)
    f_rows = summarise(f_steps, false_seqs)
    print(f"  scored {len(f_steps)} coupling steps")

    print(f"TRUE peptides: {len(true_seqs)}")
    t_steps = score(true_seqs)
    t_rows = summarise(t_steps, true_seqs)
    print(f"  scored {len(t_steps)} coupling steps")

    for r in f_rows:
        r["synthesis_flag"] = "FALSE"
    for r in t_rows:
        r["synthesis_flag"] = "TRUE"
    allrows = f_rows + t_rows

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    detail = os.path.join(OUT_DIR, f"peptimizer_report_{stamp}.csv")
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        cols = ["sequence", "synthesis_flag", "length", "n_steps",
                "min_first_diff", "mean_first_diff", "n_steps_below_-0.4"]
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in allrows)

    def stats(rows, key):
        v = np.array([r[key] for r in rows])
        return f"mean {v.mean():+.4f}  sd {v.std():.4f}  min {v.min():+.4f}  max {v.max():+.4f}"

    labels = [1 if r["synthesis_flag"] == "FALSE" else 0 for r in allrows]
    # more negative first_diff = more aggregation = expect FALSE, so negate for AUC
    auc_min = auc_roc([-r["min_first_diff"] for r in allrows], labels)
    auc_mean = auc_roc([-r["mean_first_diff"] for r in allrows], labels)
    auc_cnt = auc_roc([r["n_steps_below_-0.4"] for r in allrows], labels)

    # Length is a confounder: min over more coupling steps is lower by construction, and
    # the FALSE class skews long (87% are 15-mers). Report length's own AUC, and re-check
    # inside the 15-mer stratum where n_steps is identical for every peptide.
    auc_len = auc_roc([r["length"] for r in allrows], labels)
    s15 = [r for r in allrows if r["length"] == 15]
    lab15 = [1 if r["synthesis_flag"] == "FALSE" else 0 for r in s15]
    auc_min15 = auc_roc([-r["min_first_diff"] for r in s15], lab15)
    auc_mean15 = auc_roc([-r["mean_first_diff"] for r in s15], lab15)
    n_f15 = sum(lab15)

    L = ["peptimizer (fast-flow synthesis) - pretrained 'minimal' model",
         f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
         "learningmatter-mit/peptimizer, model/model_synthesis/synthesis_minimal.hdf5",
         "Mohapatra et al., ACS Cent. Sci. 2020, doi:10.1021/acscentsci.0c00979",
         f"Source: data/raw/peptide_baza.csv",
         "",
         "Label: first_diff = deprotection peak height - width.",
         "MORE NEGATIVE = more aggregation = harder coupling.",
         "Per peptide: min = worst coupling step, mean = average over steps.",
         "",
         f"FALSE peptides (synthesis failed)   : n={len(f_rows)}, "
         f"{sum(r['n_steps'] for r in f_rows)} coupling steps",
         f"  min_first_diff : {stats(f_rows, 'min_first_diff')}",
         f"  mean_first_diff: {stats(f_rows, 'mean_first_diff')}",
         "",
         f"TRUE peptides (synthesis succeeded) : n={len(t_rows)}, "
         f"{sum(r['n_steps'] for r in t_rows)} coupling steps",
         f"  min_first_diff : {stats(t_rows, 'min_first_diff')}",
         f"  mean_first_diff: {stats(t_rows, 'mean_first_diff')}",
         "",
         "== Discrimination (positive = FALSE / will fail) ==",
         f"  AUC-ROC, -min_first_diff      : {auc_min:.3f}   (0.500 = chance)",
         f"  AUC-ROC, -mean_first_diff     : {auc_mean:.3f}",
         f"  AUC-ROC, n_steps_below_-0.4   : {auc_cnt:.3f}",
         "",
         "== Length confound control ==",
         "min over more coupling steps is lower by construction, and the FALSE class",
         "skews long, so length alone carries signal. Re-checked in the 15-mer stratum,",
         "where every peptide has the same number of steps:",
         f"  AUC-ROC, length alone         : {auc_len:.3f}",
         f"  15-mers only: n={len(s15)} ({n_f15} FALSE / {len(s15)-n_f15} TRUE)",
         f"    AUC-ROC, -min_first_diff    : {auc_min15:.3f}   <- survives the control",
         f"    AUC-ROC, -mean_first_diff   : {auc_mean15:.3f}",
         "",
         "CAVEAT: the model was trained on automated fast-flow synthesis (AFPS) at MIT;",
         "this dataset is batch Fmoc SPPS from Gutman et al. 2022. Different instrument",
         "and conditions, so transfer is an assumption, not a given.",
         "",
         f"Per-peptide detail: {os.path.basename(detail)}"]

    summary = os.path.join(OUT_DIR, f"peptimizer_report_{stamp}.txt")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print()
    print("\n".join(L))
    print()
    print("wrote:", detail)
    print("wrote:", summary)


if __name__ == "__main__":
    main()
