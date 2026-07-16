"""Run PepFuNN's empirical synthesis/solubility rules over the FALSE-synthesis peptides.

Run from the project root:

    python experiments/predictors/pepfunn_test.py

Setup (PepFuNN is not vendored - see .gitignore):

    git clone https://github.com/novonordisk-research/pepfunn.git experiments/predictors/pepfunn
    pip install rdkit biopython urllib3 requests
    pip install -e experiments/predictors/pepfunn --no-deps

`--no-deps` skips igraph/networkx/seaborn/sphinx-jsonschema, which only PepFuNN's
clustering modules need; pepfunn.sequence needs just rdkit + biopython + numpy.

Compatibility: PepFuNN declares `biopython >= 1.79` but its code calls
ProteinAnalysis.get_amino_acids_percent(), removed in newer BioPython (1.87 exposes
the `amino_acids_percent` property instead). Pinning an old BioPython is not an
option on Python 3.13 (no wheels, source build fails), so the shim below restores
the old name and the vendored clone stays byte-identical to upstream.

Caveat on interpretation: every input here is a known synthesis FAILURE, so these
counts have no negative control. A discard rate is only meaningful next to the rate
on the TRUE (successful) peptides.
"""
import collections
import csv
import datetime
import glob
import os

# --- compat shim: must run before importing pepfunn ---
from Bio.SeqUtils.ProtParam import ProteinAnalysis

if not hasattr(ProteinAnalysis, "get_amino_acids_percent"):
    ProteinAnalysis.get_amino_acids_percent = lambda self: self.amino_acids_percent

from pepfunn.sequence import Sequence  # noqa: E402

CSV_SRC = os.path.join("data", "raw", "peptide_baza.csv")
OUT_DIR = os.path.join("experiments", "predictors")

# Rule text below describes what the CODE does. PepFuNN's own docstrings disagree
# in two places: rule 6 is documented as "2 prolines" but the regex needs 3+, and
# rule 8 is documented as "ends with N or Q" but the regex ^[NQ] matches the start.
SYN_RULES = {
    6: "3+ consecutive prolines  [P]{3,}   (docstring says '2 prolines')",
    7: "motif DG or DP present   D[GP]",
    8: "starts with N or Q       ^[NQ]     (docstring says 'ends with')",
    9: "a run of >=5 residues with no charged residue (H,R,K,D,E)",
    10: "oxidation-sensitive residue present (M, C or W)",
}
SOL_RULES = {
    1: "hydrophobic+charged residues > 45% of length",
    2: "net charge at pH 7 > +1",
    3: "more than one Gly or Pro",
    4: "first or last residue is charged",
    5: "any residue >= 30% of the sequence",
}


def read_false_sequences(path=CSV_SRC):
    with open(path, newline="", encoding="utf-8") as fh:
        return [r["peptide_seq"].strip()
                for r in csv.DictReader(fh, delimiter=";")
                if r["synthesis_flag"].strip().upper() == "FALSE"]


def analyse(seqs):
    rows = []
    for i, q in enumerate(seqs, 1):
        s = Sequence(q)
        syn_ids = [6 + j for j, v in enumerate(s.set_syn_rules) if v]
        sol_ids = [1 + j for j, v in enumerate(s.set_sol_rules) if v]
        rows.append({
            "sequence": q,
            "length": len(q),
            "syn_failed": s.synthesis_rules_failed,
            "syn_rules": "|".join(map(str, syn_ids)) or "none",
            "sol_failed": s.solubility_rules_failed,
            "sol_rules": "|".join(map(str, sol_ids)) or "none",
            "mol_weight": round(s.mol_weight, 2),
            "net_charge": round(s.netCharge, 2),
            "avg_hydro": round(s.avg_hydro, 3),
            "isoelectric_point": round(float(s.isoelectric_point), 2),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(seqs)}")
    return rows


def build_summary(rows, detail_name):
    n = len(rows)
    syn_dist = collections.Counter(r["syn_failed"] for r in rows)
    sol_dist = collections.Counter(r["sol_failed"] for r in rows)
    syn_hits = collections.Counter()
    for r in rows:
        if r["syn_rules"] != "none":
            for rid in r["syn_rules"].split("|"):
                syn_hits[int(rid)] += 1
    rejected = sum(1 for r in rows if r["syn_failed"] > 1)  # pepfunn library.py criterion

    L = ["PepFuNN empirical rules - FALSE-synthesis peptides",
         f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
         "PepFuNN 1.0.0 (novonordisk-research/pepfunn), module pepfunn.sequence",
         f"Source: {CSV_SRC}  (synthesis_flag = FALSE, n={n})",
         "",
         "== Synthesis rules failed per peptide (0-5) =="]
    for k in sorted(syn_dist):
        L.append(f"  {k} rule(s): {syn_dist[k]:3d}  ({syn_dist[k]/n*100:5.1f}%)")
    L.append(f"  PepFuNN's own reject criterion (>1 rule failed): {rejected}/{n} "
             f"({rejected/n*100:.1f}%) would be discarded")
    L += ["", "== How often each synthesis rule fires =="]
    for rid in sorted(SYN_RULES):
        c = syn_hits.get(rid, 0)
        L.append(f"  rule {rid:2d}: {c:3d} ({c/n*100:5.1f}%)  {SYN_RULES[rid]}")
    L += ["", "== Solubility rules failed per peptide (0-5) =="]
    for k in sorted(sol_dist):
        L.append(f"  {k} rule(s): {sol_dist[k]:3d}  ({sol_dist[k]/n*100:5.1f}%)")
    L.append("")

    # cross-tab against the Excel difficult-coupling predictor, if a report exists
    prev = sorted(glob.glob(os.path.join(OUT_DIR, "difficulty_report_*.csv")))
    if prev:
        with open(prev[-1], newline="", encoding="utf-8") as fh:
            excel = {r["sequence"]: r["category"] for r in csv.DictReader(fh, delimiter=";")}
        ct = collections.Counter()
        for r in rows:
            cat = excel.get(r["sequence"])
            if cat:
                ct[(cat, r["syn_failed"] > 1)] += 1
        L.append(f"== Cross-tab vs Excel predictor ({os.path.basename(prev[-1])}) ==")
        L.append("   Excel category   PepFuNN discard(>1)   PepFuNN keep(<=1)   total")
        for cat in ("RED", "GREEN", "NONE"):
            d, k = ct[(cat, True)], ct[(cat, False)]
            if d + k:
                L.append(f"   {cat:14} {d:16d} {k:19d} {d+k:7d}")
        agree = ct[("RED", True)] + ct[("GREEN", False)] + ct[("NONE", False)]
        L.append(f"   Rough agreement (RED~discard, GREEN/NONE~keep): {agree}/{n} ({agree/n*100:.1f}%)")
        L.append("")
    L.append("NOTE: all inputs are known synthesis failures - there is no negative")
    L.append("control here. Compare against the TRUE peptides before drawing conclusions.")
    L.append("")
    L.append(f"Per-sequence detail: {detail_name}")
    return L


def main():
    seqs = read_false_sequences()
    print(f"running PepFuNN over {len(seqs)} FALSE peptides ...")
    rows = analyse(seqs)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    detail = os.path.join(OUT_DIR, f"pepfunn_report_{stamp}.csv")
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)

    lines = build_summary(rows, os.path.basename(detail))
    summary = os.path.join(OUT_DIR, f"pepfunn_report_{stamp}.txt")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print("wrote:", detail)
    print("wrote:", summary)


if __name__ == "__main__":
    main()
