"""Evaluate PepFuNN's empirical rules as a synthesis-outcome predictor.

Runs PepFuNN over BOTH classes (TRUE = synthesis succeeded, FALSE = failed) so the
rules can be scored as a classifier instead of just described. Without the TRUE
peptides as a negative control, a discard rate on the FALSE set means nothing.

Run from the project root (see pepfunn_test.py docstring for PepFuNN setup):

    python experiments/predictors/pepfunn_evaluate.py

Framing: PepFuNN's library.py discards a peptide when synthesis_rules_failed > 1.
Read as a prediction, "discard" = "this peptide will fail synthesis", so the
positive class is FALSE (synthesis_flag = FALSE).
"""
import collections
import csv
import datetime
import os

# --- compat shim: must run before importing pepfunn (see pepfunn_test.py) ---
from Bio.SeqUtils.ProtParam import ProteinAnalysis

if not hasattr(ProteinAnalysis, "get_amino_acids_percent"):
    ProteinAnalysis.get_amino_acids_percent = lambda self: self.amino_acids_percent

from pepfunn.sequence import Sequence  # noqa: E402

CSV_SRC = os.path.join("data", "raw", "peptide_baza.csv")
OUT_DIR = os.path.join("experiments", "predictors")
DISCARD_THRESHOLD = 1  # pepfunn library.py: discard when syn_failed > 1

SYN_RULES = {
    6: "3+ consecutive prolines [P]{3,}",
    7: "motif DG or DP present",
    8: "starts with N or Q",
    9: "run of >=5 residues with no charged residue",
    10: "oxidation-sensitive residue (M/C/W)",
}


def read_dataset(path=CSV_SRC):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [(r["peptide_seq"].strip(), r["synthesis_flag"].strip().upper())
                for r in csv.DictReader(fh, delimiter=";")]
    return [(s, f) for s, f in rows if f in ("TRUE", "FALSE")]


def analyse(data):
    out = []
    for i, (seq, flag) in enumerate(data, 1):
        s = Sequence(seq)
        out.append({
            "sequence": seq,
            "synthesis_flag": flag,
            "length": len(seq),
            "syn_failed": s.synthesis_rules_failed,
            "syn_rules": "|".join(str(6 + j) for j, v in enumerate(s.set_syn_rules) if v) or "none",
            "sol_failed": s.solubility_rules_failed,
            "sol_rules": "|".join(str(1 + j) for j, v in enumerate(s.set_sol_rules) if v) or "none",
            "discard": int(s.synthesis_rules_failed > DISCARD_THRESHOLD),
        })
        if i % 200 == 0:
            print(f"  {i}/{len(data)}")
    return out


def auc_roc(scores, labels):
    """AUC via the rank/Mann-Whitney identity, with tie correction."""
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
    n_true = sum(1 for _, f in data if f == "TRUE")
    n_false = sum(1 for _, f in data if f == "FALSE")
    print(f"running PepFuNN over {len(data)} peptides ({n_true} TRUE / {n_false} FALSE) ...")
    rows = analyse(data)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    detail = os.path.join(OUT_DIR, f"pepfunn_evaluate_{stamp}.csv")
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)

    false_rows = [r for r in rows if r["synthesis_flag"] == "FALSE"]
    true_rows = [r for r in rows if r["synthesis_flag"] == "TRUE"]

    # confusion matrix: positive class = FALSE (will fail), prediction = discard
    tp = sum(r["discard"] for r in false_rows)
    fn = len(false_rows) - tp
    fp = sum(r["discard"] for r in true_rows)
    tn = len(true_rows) - fp
    n = len(rows)

    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / n
    bal = (sens + spec) / 2
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    prevalence = len(false_rows) / n

    labels = [1 if r["synthesis_flag"] == "FALSE" else 0 for r in rows]
    auc_syn = auc_roc([r["syn_failed"] for r in rows], labels)
    auc_sol = auc_roc([r["sol_failed"] for r in rows], labels)

    L = ["PepFuNN as a synthesis-outcome predictor - both classes",
         f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
         "PepFuNN 1.0.0 (novonordisk-research/pepfunn), module pepfunn.sequence",
         f"Source: {CSV_SRC}",
         f"Peptides: {n}  ({len(true_rows)} TRUE / {len(false_rows)} FALSE, "
         f"{prevalence*100:.1f}% FALSE)",
         "",
         "== Synthesis rules failed: TRUE vs FALSE ==",
         "  rules   TRUE            FALSE"]
    td = collections.Counter(r["syn_failed"] for r in true_rows)
    fd = collections.Counter(r["syn_failed"] for r in false_rows)
    for k in sorted(set(td) | set(fd)):
        L.append(f"    {k}    {td[k]:5d} ({td[k]/len(true_rows)*100:5.1f}%)  "
                 f"{fd[k]:5d} ({fd[k]/len(false_rows)*100:5.1f}%)")
    L += ["",
          f"  mean rules failed: TRUE {sum(r['syn_failed'] for r in true_rows)/len(true_rows):.3f}"
          f"  |  FALSE {sum(r['syn_failed'] for r in false_rows)/len(false_rows):.3f}",
          "",
          "== Discard rate (PepFuNN criterion: syn_failed > 1) ==",
          f"  TRUE  (synthesis succeeded): {fp:4d}/{len(true_rows)} discarded "
          f"({fp/len(true_rows)*100:.1f}%)  <- false alarms",
          f"  FALSE (synthesis failed)   : {tp:4d}/{len(false_rows)} discarded "
          f"({tp/len(false_rows)*100:.1f}%)  <- caught",
          f"  difference: {tp/len(false_rows)*100 - fp/len(true_rows)*100:+.1f} pp",
          "",
          "== Confusion matrix (positive = FALSE / will fail) ==",
          f"                  predicted discard   predicted keep",
          f"  actual FALSE    {tp:12d}   {fn:16d}",
          f"  actual TRUE     {fp:12d}   {tn:16d}",
          "",
          "== Metrics ==",
          f"  sensitivity (recall on FALSE): {sens:.3f}",
          f"  specificity (on TRUE)        : {spec:.3f}",
          f"  precision (PPV)              : {prec:.3f}   (base rate {prevalence:.3f})",
          f"  accuracy                     : {acc:.3f}",
          f"  balanced accuracy            : {bal:.3f}   (0.500 = chance)",
          f"  MCC                          : {mcc:+.3f}   (0.000 = chance)",
          f"  AUC-ROC, syn_failed as score : {auc_syn:.3f}   (0.500 = chance)",
          f"  AUC-ROC, sol_failed as score : {auc_sol:.3f}",
          "",
          "== Per-rule fire rate: TRUE vs FALSE ==",
          "  rule   TRUE     FALSE    diff    description"]
    for rid in sorted(SYN_RULES):
        t = sum(1 for r in true_rows if str(rid) in r["syn_rules"].split("|"))
        f_ = sum(1 for r in false_rows if str(rid) in r["syn_rules"].split("|"))
        tp_, fp_ = t / len(true_rows) * 100, f_ / len(false_rows) * 100
        L.append(f"  {rid:4d}  {tp_:5.1f}%   {fp_:5.1f}%  {fp_-tp_:+6.1f}pp  {SYN_RULES[rid]}")
    L += ["", f"Per-peptide detail: {os.path.basename(detail)}"]

    summary = os.path.join(OUT_DIR, f"pepfunn_evaluate_{stamp}.txt")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print()
    print("\n".join(L))
    print()
    print("wrote:", detail)
    print("wrote:", summary)


if __name__ == "__main__":
    main()
