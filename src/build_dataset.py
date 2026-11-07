#!/usr/bin/env python3
"""
Build the peptide feature table directly from the source spreadsheet.

Replaces src/ingest_excel_to_db.py, which routed through MySQL. That indirection was the
reason the repository carried a hardcoded database password, and the reason the committed
script and the committed data drifted apart. This script needs no database and no
credential: one command, one output file, reproducible.

    python src/build_dataset.py                 # build data/processed/peptide_features.csv
    python src/build_dataset.py --legacy-check  # verify we reproduce data/raw/peptide_baza.csv
    python src/build_dataset.py --convention biochem --with-fractions

Run from the project root.

WHAT CHANGED vs. the old pipeline
---------------------------------
1. Compositional columns are named for what they contain. The old code built a list of
   [count, fraction] per group -- 18 values for 9 groups -- and wrote the FIRST NINE into
   nine differently-named columns, so `small_group` held the fraction of Tiny,
   `aliphatic_group` held the count of Small, and so on. Eight of nine columns held a
   different quantity than their name, and polar/charged/basic/acidic counts were absent
   from the dataset entirely. Verified: the hypothesis reproduces all 15,939 historical
   cells exactly. See --legacy-check.

2. `hydrophobic_cornette` is dropped. It was constant 0.0 in every row, because the scale
   is not available in peptides==0.5.0 and the old code substituted None. A constant
   column is not a feature.

3. Positional and motif features are added. The old feature set was entirely composition
   and whole-sequence means, which discard the position information that drives on-resin
   aggregation. See src/sequence_features.py.

4. Counts only by default. `frac_x == n_x / peptide_len` exactly, and peptide_len is a
   column, so emitting both families injects near-perfect collinearity into feature
   selection. --with-fractions restores them if you want them.
"""

import argparse
import csv
import os
import sys

from openpyxl import load_workbook

try:
    import peptides
except ImportError:
    sys.exit("The 'peptides' package is required: pip install peptides==0.5.0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence_features as sf  # noqa: E402

XLSX = os.path.join('data', 'raw', 'ao2c02425_si_002.xlsx')
LEGACY_CSV = os.path.join('data', 'raw', 'peptide_baza.csv')
OUT_CSV = os.path.join('data', 'processed', 'peptide_features.csv')

# Scales available in peptides==0.5.0. Cornette is NOT among them.
HYDRO_SCALES = ['Janin', 'Engelman', 'KyteDoolittle', 'HoppWoods', 'Eisenberg', 'Roseman']


def physicochemical(seq):
    """Descriptors from the `peptides` library. Values depend on the library version;
    requirements.txt pins peptides==0.5.0 for this reason."""
    p = peptides.Peptide(seq)
    cru = p.cruciani_properties()
    out = {
        'cruciani_prp1': cru[0],
        'cruciani_prp2': cru[1],
        'cruciani_prp3': cru[2],
        'instability_index': p.instability_index(),
        'boman': p.boman(),
        'hydrophobic_moment': p.hydrophobic_moment(angle=100, window=min(len(seq), 11)),
        'aliphatic_index': p.aliphatic_index(),
        'isoelectric_point': p.isoelectric_point('Lehninger'),
        'charge': p.charge(pKscale='Lehninger'),
    }
    for s in HYDRO_SCALES:
        out['hydrophobic_' + s.lower()] = p.hydrophobicity(s)
    return out


def read_source():
    """Yield (sequence, label) from sheet S2 of the supporting-information workbook.

    Row 1 is the table caption, row 2 the header ('peptide', 'label'); data starts row 3.
    """
    if not os.path.exists(XLSX):
        sys.exit('Source spreadsheet not found: %s (run from the project root)' % XLSX)
    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb['S2'] if 'S2' in wb.sheetnames else wb.active
    out = []
    for row in ws.iter_rows(min_row=3, values_only=True):
        if not row or row[0] is None:
            continue
        seq = str(row[0]).strip().upper()
        if not seq:
            continue
        bad = sorted(set(seq) - sf.STANDARD_AA)
        if bad:
            print('  skipping %r: non-standard residues %s' % (seq, bad), file=sys.stderr)
            continue
        out.append((seq, int(row[1])))
    wb.close()
    return out


# ------------------------------------------------------------------ legacy check

LEGACY_COMPOSITIONAL = ['tiny_group', 'small_group', 'aliphatic_group', 'aromatic_group',
                        'non-polar_group', 'polar_group', 'charged_group', 'basic_group',
                        'acidic_group']


def legacy_row(seq):
    """Reproduce the historical peptide_baza.csv row, bug and all.

    Two faults are reproduced deliberately:
      * compositional() returned [count, fraction] per group (18 values) and only the
        first nine were written, into nine differently-named columns
      * the Aliphatic group had no methionine; src/ingest_excel_to_db.py later added M
        and the CSV was never regenerated, so 468 rows disagree with that script
    """
    comp = []
    for _, group in sf.GROUPS_LEGACY:
        c = sum(seq.count(a) for a in group)
        comp.append(float(c))
        comp.append(c / len(seq))
    row = dict(zip(LEGACY_COMPOSITIONAL, comp[:9]))
    phy = physicochemical(seq)
    row.update({
        'peptide_len': len(seq),
        'hydrophobic_janin': phy['hydrophobic_janin'],
        'hydrophobic_engleman': phy['hydrophobic_engelman'],
        'hydrophobic_moment': phy['hydrophobic_moment'],
        'aliphatic_index': phy['aliphatic_index'],
        'isoelectric_point': phy['isoelectric_point'],
        'charge': phy['charge'],
        'cruciani_prp1': phy['cruciani_prp1'],
        'cruciani_prp2': phy['cruciani_prp2'],
        'cruciani_prp3': phy['cruciani_prp3'],
        'instability_index': phy['instability_index'],
        'boman': phy['boman'],
        'hydrophobic_kyte-doolittle': phy['hydrophobic_kytedoolittle'],
        'hydrophobic_hopp-woods': phy['hydrophobic_hoppwoods'],
        'hydrophobic_cornette': 0.0,
        'hydrophobic_eisenberg': phy['hydrophobic_eisenberg'],
        'hydrophobic_roseman': phy['hydrophobic_roseman'],
    })
    return row


def legacy_check(tol=1e-9):
    """Rebuild every historical value from the source xlsx and diff against the committed
    CSV. This is the evidence that the new code understands the old data before replacing
    it: if this does not pass, nothing else here should be trusted."""
    if not os.path.exists(LEGACY_CSV):
        sys.exit('Legacy CSV not found: %s' % LEGACY_CSV)
    with open(LEGACY_CSV, newline='', encoding='utf-8') as fh:
        old = list(csv.DictReader(fh, delimiter=';'))
    print('legacy rows: %d' % len(old))

    checked = mismatch = 0
    bycol = {}
    for r in old:
        seq = r['peptide_seq'].strip()
        want = legacy_row(seq)
        for col, v in want.items():
            if col not in r or r[col] in ('', None):
                continue
            checked += 1
            if abs(float(r[col]) - float(v)) > tol:
                mismatch += 1
                bycol[col] = bycol.get(col, 0) + 1
    print('cells compared : %d' % checked)
    print('mismatches     : %d' % mismatch)
    if bycol:
        for c, k in sorted(bycol.items(), key=lambda kv: -kv[1]):
            print('   %-30s %d' % (c, k))
        print('\nFAIL - the rebuild does not reproduce the committed data.')
        return 1
    print('\nPASS - every historical value is reproduced exactly from the source xlsx,')
    print('including the compositional mis-mapping and the methionine-free Aliphatic group.')
    return 0


# ----------------------------------------------------------------------- build

def build(convention, fractions, out_path):
    rows = read_source()
    print('read %d peptides from %s' % (len(rows), XLSX))
    records = []
    for i, (seq, label) in enumerate(rows, start=1):
        rec = {'id': i, 'peptide_seq': seq}
        rec.update(sf.all_features(seq, convention=convention, fractions=fractions))
        rec.update(physicochemical(seq))
        rec['targetcol'] = label
        records.append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = list(records[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=';')
        w.writeheader()
        w.writerows(records)

    pos = sum(r['targetcol'] for r in records)
    print('wrote %s' % out_path)
    print('  rows     : %d' % len(records))
    print('  columns  : %d' % len(cols))
    print('  label    : %d success / %d failure (%.1f%% minority)'
          % (pos, len(records) - pos, 100.0 * (len(records) - pos) / len(records)))
    print('  convention: %s%s' % (convention, ' (+fractions)' if fractions else ''))
    lengths = sorted(r['peptide_len'] for r in records)
    print('  lengths  : %d-%d, median %d' % (lengths[0], lengths[-1], lengths[len(lengths) // 2]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--legacy-check', action='store_true',
                    help='verify this code reproduces data/raw/peptide_baza.csv exactly, '
                         'then exit')
    ap.add_argument('--convention', choices=('paper', 'biochem'), default='paper',
                    help="residue-group convention. 'paper' matches Gutman et al. 2022 so "
                         "results compare directly to its published AUC table; 'biochem' "
                         "uses standard textbook groupings. Default: paper")
    ap.add_argument('--with-fractions', action='store_true',
                    help='also emit frac_* columns. They equal n_*/peptide_len exactly, so '
                         'this injects collinearity; off by default')
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    if args.legacy_check:
        return legacy_check()
    return build(args.convention, args.with_fractions, args.out)


if __name__ == '__main__':
    sys.exit(main())
