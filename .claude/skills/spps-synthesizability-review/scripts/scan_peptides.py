#!/usr/bin/env python3
"""
Scan peptide sequences for Fmoc-SPPS synthesis liabilities.

Deterministic rule engine. Every rule is sequence-computable and carries a source.
Reads CSV/TSV/XLSX/FASTA/plain-text, finds the sequence column, emits findings.

Usage:
    python scan_peptides.py <file> [--column NAME] [--sep ';'] [--format text|json|csv]
                                   [--top N] [--min-severity LOW] [--summary-only]

Exit codes: 0 = scanned, 2 = could not find sequences.

Rule sources are listed in ../references/liability-rules.md. This reports MECHANISTIC
RISK FLAGS, not a calibrated probability of failure. See "Honest limits" in SKILL.md.
"""

import argparse
import csv as _csv
import json
import os
import re
import sys
from collections import Counter, OrderedDict

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte & Doolittle 1982, J Mol Biol 157:105 - hydropathy index
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Janin 1979, Nature 277:491 - free energy of transfer (interior/surface)
JANIN = {
    'A': 0.3, 'R': -1.4, 'N': -0.5, 'D': -0.6, 'C': 0.9, 'Q': -0.7, 'E': -0.7,
    'G': 0.3, 'H': -0.1, 'I': 0.7, 'L': 0.5, 'K': -1.8, 'M': 0.4, 'F': 0.5,
    'P': -0.3, 'S': -0.1, 'T': -0.2, 'W': 0.3, 'Y': -0.4, 'V': 0.6,
}

ALIPHATIC_H = set("ILMV")      # Gutman et al. 'h'
BETA_BRANCHED = set("IVT")     # slow coupling, steric
PSEUDOPRO_PRECURSOR = set("STC")  # pseudoproline building blocks derive from Ser/Thr/Cys

# Asp-Xaa aspartimide sensitivity (Gyros Protein Technologies, SPPS Tips For Success)
ASPARTIMIDE = OrderedDict([
    ('G', ('CRITICAL', '+++++')),
    ('N', ('HIGH', '+++')),
    ('D', ('MEDIUM', '++')),
    ('S', ('MEDIUM', '++')),
    ('T', ('MEDIUM', '++')),
    ('C', ('MEDIUM', '++')),
    ('A', ('LOW', '+')),
    ('Q', ('LOW', '+')),
    ('H', ('LOW', '+ (pH dependent)')),
])

SEV_ORDER = {'INFO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}

# Two categories, because they answer different questions.
#   ASSEMBLY - predicts whether the chain will assemble cleanly. Empirically enriches
#              for failure on the Gutman 2022 dataset (n=1771, 13.1% base failure rate).
#   PROCESS  - real chemistry needing a specific mitigation, but NOT predictive of crude
#              purity on that dataset. Treat as a protocol checklist, not a risk score.
#
# 'lift' = P(failure | rule fires) / P(failure | rule does not fire), measured on
# data/raw/peptide_baza.csv. Lift ~1.0 means no discriminative power. Below 1.0 means
# the rule fires MORE often on successful syntheses - see the Arg note in SKILL.md.
RULE_META = {
    'agg.hydrophobic_window':  ('ASSEMBLY', 3.19),
    'agg.h_stretch':           ('ASSEMBLY', 2.16),
    'agg.beta_branched_run':   ('ASSEMBLY', 1.93),
    # Length is reported but NOT trusted: within the homogeneous length-15 cohort that
    # holds 87% of this dataset's failures, length AUC is 0.500 by construction, and the
    # >=20 rule fires more often on successes. See "Confounds" in SKILL.md.
    'agg.length':              ('ASSEMBLY', 0.52),
    'agg.pseudoproline_site':  ('ASSEMBLY', None),
    'side.diketopiperazine':   ('PROCESS', 1.46),
    'side.aspartimide':        ('PROCESS', 1.37),
    'side.pyroglutamate':      ('PROCESS', 1.16),
    'side.oxidation':          ('PROCESS', 1.15),
    'side.racemization':       ('PROCESS', 0.91),
    'side.arg_lactam':         ('PROCESS', 0.58),
    'side.disulfide':          ('PROCESS', None),
    'stability.deamidation':   ('PROCESS', 0.39),
    'seq.nonstandard':         ('DATA', None),
    'seq.empty':               ('DATA', None),
}


def rule_category(rule):
    return RULE_META.get(rule, ('PROCESS', None))[0]


class Finding:
    __slots__ = ('rule', 'severity', 'message', 'positions')

    def __init__(self, rule, severity, message, positions=None):
        self.rule = rule
        self.severity = severity
        self.message = message
        self.positions = positions or []

    def as_dict(self):
        return {'rule': self.rule, 'severity': self.severity,
                'message': self.message, 'positions': self.positions}


# ---------------------------------------------------------------- rules

def longest_run(seq, members):
    """Return (length, start_index_1based) of the longest run of residues in `members`."""
    best = best_start = 0
    cur = cur_start = 0
    for i, aa in enumerate(seq):
        if aa in members:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    return best, best_start + 1


def window_means(seq, scale, w=5):
    """Mean scale value over every window of width w. Returns list of (start_1based, mean)."""
    if len(seq) < w:
        vals = [scale.get(a, 0.0) for a in seq]
        return [(1, sum(vals) / len(vals))] if vals else []
    out = []
    for i in range(len(seq) - w + 1):
        chunk = seq[i:i + w]
        out.append((i + 1, sum(scale.get(a, 0.0) for a in chunk) / w))
    return out


def check_validity(seq):
    f = []
    if not seq:
        f.append(Finding('seq.empty', 'CRITICAL', 'Empty sequence'))
        return f
    bad = sorted({a for a in seq if a not in STANDARD_AA})
    if bad:
        f.append(Finding('seq.nonstandard', 'HIGH',
                         'Non-standard residue letters: %s. Descriptor scales are undefined '
                         'for these; they silently contribute 0 to every hydrophobicity mean.'
                         % ', '.join(bad)))
    return f


def check_aggregation(seq):
    """Chain-assembly difficulty: beta-sheet association makes termini inaccessible."""
    f = []
    n = len(seq)

    run, pos = longest_run(seq, ALIPHATIC_H)
    if run >= 5:
        f.append(Finding('agg.h_stretch', 'CRITICAL',
                         'Aliphatic-hydrophobic run of %d (I/L/M/V) at position %d. Strong '
                         'on-resin aggregation risk; expect incomplete couplings through '
                         'this region.' % (run, pos), [pos]))
    elif run >= 3:
        f.append(Finding('agg.h_stretch', 'HIGH' if run == 4 else 'MEDIUM',
                         'Aliphatic-hydrophobic run of %d (I/L/M/V) at position %d.'
                         % (run, pos), [pos]))

    bb, bpos = longest_run(seq, BETA_BRANCHED)
    if bb >= 3:
        f.append(Finding('agg.beta_branched_run', 'HIGH',
                         'Run of %d beta-branched residues (I/V/T) at position %d. Sterically '
                         'hindered couplings; extended coupling time or double coupling '
                         'indicated.' % (bb, bpos), [bpos]))

    wins = window_means(seq, KD, 5)
    if wins:
        start, mx = max(wins, key=lambda t: t[1])
        if mx >= 2.5:
            f.append(Finding('agg.hydrophobic_window', 'HIGH',
                             'Hydrophobic 5-residue window (mean Kyte-Doolittle %.2f) at '
                             'position %d: %s' % (mx, start, seq[start - 1:start + 4]),
                             [start]))
        elif mx >= 1.8:
            f.append(Finding('agg.hydrophobic_window', 'MEDIUM',
                             'Moderately hydrophobic window (mean KD %.2f) at position %d: %s'
                             % (mx, start, seq[start - 1:start + 4]), [start]))

    if n >= 30:
        f.append(Finding('agg.length', 'HIGH',
                         'Length %d. Cumulative stepwise yield dominates at this length; '
                         'consider fragment condensation or ligation.' % n))
    elif n >= 20:
        f.append(Finding('agg.length', 'MEDIUM',
                         'Length %d. Aggregation risk rises with chain length.' % n))

    # Rescue opportunity, not a liability.
    if run >= 3:
        lo, hi = max(1, pos - 2), min(n, pos + run + 1)
        sites = [i for i in range(lo, hi + 1) if seq[i - 1] in PSEUDOPRO_PRECURSOR]
        if sites:
            f.append(Finding('agg.pseudoproline_site', 'INFO',
                             'Ser/Thr/Cys at position(s) %s sits in or beside the difficult '
                             'region - a pseudoproline dipeptide can be installed here to '
                             'disrupt beta-sheet association.'
                             % ', '.join(map(str, sites)), sites))
    return f


def check_side_reactions(seq):
    f = []
    n = len(seq)

    # Aspartimide: Asp-Xaa, graded by the following residue.
    for i in range(n - 1):
        if seq[i] == 'D':
            nxt = seq[i + 1]
            if nxt in ASPARTIMIDE:
                sev, mark = ASPARTIMIDE[nxt]
                f.append(Finding('side.aspartimide', sev,
                                 'Asp-%s at position %d (sensitivity %s). Repeated piperidine '
                                 'exposure drives aspartimide formation and its by-products.'
                                 % (nxt, i + 1, mark), [i + 1]))

    # Pyroglutamate from N-terminal Gln.
    if seq and seq[0] == 'Q':
        f.append(Finding('side.pyroglutamate', 'HIGH',
                         'N-terminal Gln. Cyclises to pyroglutamate, capping the chain.', [1]))
    elif seq and seq[0] == 'E':
        f.append(Finding('side.pyroglutamate', 'LOW',
                         'N-terminal Glu. Lower but non-zero pyroglutamate risk.', [1]))

    # Diketopiperazine: C-terminal dipeptide cyclises and cleaves the chain off the resin.
    if n >= 2:
        c1, c2 = seq[-1], seq[-2]
        if c1 == 'P' or c2 == 'P':
            f.append(Finding('side.diketopiperazine', 'HIGH',
                             'Pro at C-terminal position %d. DKP formation cleaves the peptide '
                             'from the resin during Fmoc removal; 2-chlorotrityl resin is the '
                             'standard mitigation.' % (n if c1 == 'P' else n - 1),
                             [n if c1 == 'P' else n - 1]))
        elif 'G' in (c1, c2):
            f.append(Finding('side.diketopiperazine', 'MEDIUM',
                             'Gly in the C-terminal dipeptide (%s%s). Flexible C-terminus '
                             'raises DKP risk.' % (c2, c1), [n - 1]))

    # Racemization-prone residues.
    for aa, label in (('H', 'His'), ('C', 'Cys')):
        pos = [i + 1 for i, a in enumerate(seq) if a == aa]
        if pos:
            f.append(Finding('side.racemization', 'MEDIUM',
                             '%s at position(s) %s. Prone to epimerisation under base-mediated '
                             'activation.' % (label, ', '.join(map(str, pos))), pos))

    # Arg: delta-lactam during activation -> des-Arg deletion peptides.
    rpos = [i + 1 for i, a in enumerate(seq) if a == 'R']
    if len(rpos) >= 2:
        f.append(Finding('side.arg_lactam', 'INFO',
                         '%d Arg residues at %s. delta-lactam formation during activation can '
                         'give des-Arg deletions; recoupling is commonly required. NOTE: on an '
                         'MS-abundance label this rule fires MORE often on SUCCESSFUL '
                         'syntheses (lift 0.58) - basic residues raise ESI response. Do not '
                         'use Arg content as a difficulty signal.'
                         % (len(rpos), ', '.join(map(str, rpos))), rpos))
    elif rpos:
        f.append(Finding('side.arg_lactam', 'INFO',
                         'Arg at position %d; watch for des-Arg deletion.' % rpos[0], rpos))

    # Cys pairs: disulfide strategy needed.
    cpos = [i + 1 for i, a in enumerate(seq) if a == 'C']
    if len(cpos) >= 2:
        f.append(Finding('side.disulfide', 'MEDIUM',
                         '%d Cys residues at %s. Orthogonal protection and a defined oxidation '
                         'order are needed.' % (len(cpos), ', '.join(map(str, cpos))), cpos))

    # Oxidation-sensitive residues affect cleavage cocktail choice.
    ox = [(i + 1, a) for i, a in enumerate(seq) if a in 'MWC']
    if ox:
        f.append(Finding('side.oxidation', 'INFO',
                         'Oxidation-sensitive residues (%s). Scavenger choice at cleavage '
                         'matters. Near-universal (fires on ~54%% of typical sets) and not '
                         'discriminative - a protocol note, not a risk signal.'
                         % ', '.join('%s%d' % (a, p) for p, a in ox)))

    # Asn-Gly: deamidation. General peptide-stability knowledge, not from the SPPS sources.
    for m in re.finditer('NG', seq):
        f.append(Finding('stability.deamidation', 'LOW',
                         'Asn-Gly at position %d. Deamidation hotspot - a storage/stability '
                         'liability rather than an SPPS one.' % (m.start() + 1),
                         [m.start() + 1]))
    return f


def scan_sequence(seq):
    seq = (seq or '').strip().upper()
    findings = check_validity(seq)
    if any(f.rule == 'seq.empty' for f in findings):
        return seq, findings
    findings += check_aggregation(seq)
    findings += check_side_reactions(seq)
    findings.sort(key=lambda f: -SEV_ORDER[f.severity])
    return seq, findings


def baseline_features(seq):
    """The two features the source paper showed actually predict (AUC 0.773 combined)."""
    if not seq:
        return {}
    jan = [JANIN.get(a, 0.0) for a in seq]
    kd = [KD.get(a, 0.0) for a in seq]
    h_run, _ = longest_run(seq, ALIPHATIC_H)
    return {
        'length': len(seq),
        'janin_mean': round(sum(jan) / len(jan), 4),
        'kd_mean': round(sum(kd) / len(kd), 4),
        'h_stretch': h_run,
    }


# ---------------------------------------------------------------- input

SEQ_COL_HINTS = ('peptide_seq', 'sequence', 'peptide', 'seq', 'aa_sequence')
SEQ_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy]{4,}$')

# Words that pass SEQ_RE but are header labels, not peptides. "peptide" is seven valid
# one-letter codes; so are "label", "data", "test", "train", "file", "nan", "na".
HEADER_WORDS = {'peptide', 'peptides', 'label', 'labels', 'seq', 'sequence', 'sequences',
                'data', 'test', 'train', 'val', 'valid', 'id', 'name', 'index', 'file',
                'nan', 'na', 'null', 'none', 'true', 'false', 'class', 'target', 'y'}


def _is_sequence(v):
    v = str(v).strip()
    return bool(SEQ_RE.match(v)) and v.lower() not in HEADER_WORDS


def _looks_like_sequences(values):
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals:
        return 0.0
    return sum(1 for v in vals if _is_sequence(v)) / len(vals)


def load_sequences(path, column=None, sep=None):
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.fa', '.fasta', '.faa'):
        seqs, cur = [], []
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('>'):
                    if cur:
                        seqs.append(''.join(cur))
                        cur = []
                elif line:
                    cur.append(line)
        if cur:
            seqs.append(''.join(cur))
        return seqs, 'FASTA'

    if ext in ('.xlsx', '.xlsm', '.xltx'):
        try:
            from openpyxl import load_workbook
        except ImportError:
            sys.exit('openpyxl required to read %s' % ext)
        wb = load_workbook(path, read_only=True, data_only=True)
        # Pass 1: pick the best (sheet, column) from a sample of rows.
        best_score, best_sheet, best_col = 0.0, None, None
        for ws in wb.worksheets:
            sample = [r for _, r in zip(range(400), ws.iter_rows(values_only=True))]
            if not sample:
                continue
            for c in range(max(len(r) for r in sample)):
                score = _looks_like_sequences([r[c] for r in sample if c < len(r)])
                if score > best_score:
                    best_score, best_sheet, best_col = score, ws.title, c
        if best_score < 0.3 or best_sheet is None:
            wb.close()
            return [], None
        # Pass 2: read EVERY row of that column, so nothing is silently truncated.
        ws = wb[best_sheet]
        seqs = []
        for row in ws.iter_rows(values_only=True):
            if best_col < len(row) and row[best_col] is not None:
                v = str(row[best_col]).strip()
                if _is_sequence(v):
                    seqs.append(v)
        wb.close()
        return seqs, '%s!col%d' % (best_sheet, best_col + 1)

    # Delimited text
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
        sample = fh.read(8192)
        fh.seek(0)
        if sep is None:
            try:
                sep = _csv.Sniffer().sniff(sample, delimiters=';,\t|').delimiter
            except Exception:
                sep = ';' if sample.count(';') > sample.count(',') else ','
        reader = _csv.reader(fh, delimiter=sep)
        rows = list(reader)

    if not rows:
        return [], None
    header, body = rows[0], rows[1:]
    header = [h.strip().strip('"') for h in header]

    idx = None
    if column:
        if column not in header:
            sys.exit('Column %r not in header: %s' % (column, header))
        idx = header.index(column)
    else:
        for hint in SEQ_COL_HINTS:
            for i, h in enumerate(header):
                if h.lower() == hint:
                    idx = i
                    break
            if idx is not None:
                break
    if idx is None:
        scores = []
        for i in range(len(header)):
            col = [r[i] for r in body[:400] if i < len(r)]
            scores.append((_looks_like_sequences(col), i))
        scores.sort(reverse=True)
        if not scores or scores[0][0] < 0.3:
            return [], None
        idx = scores[0][1]

    seqs = [r[idx].strip().strip('"') for r in body if idx < len(r) and r[idx].strip()]
    return seqs, header[idx]


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--column', help='sequence column name (else auto-detected)')
    ap.add_argument('--sep', help='delimiter for text files (else sniffed)')
    ap.add_argument('--format', choices=('text', 'json', 'csv'), default='text')
    ap.add_argument('--top', type=int, default=15, help='worst N sequences to list')
    ap.add_argument('--min-severity', default='MEDIUM',
                    choices=tuple(SEV_ORDER), help='floor for per-sequence listing')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit('No such file: %s' % args.path)

    seqs, where = load_sequences(args.path, args.column, args.sep)
    if not seqs:
        print('Could not identify a sequence column in %s' % args.path, file=sys.stderr)
        print('Pass --column NAME explicitly.', file=sys.stderr)
        return 2

    floor = SEV_ORDER[args.min_severity]
    results, rule_counts, sev_counts = [], Counter(), Counter()

    for s in seqs:
        seq, findings = scan_sequence(s)
        worst = max((SEV_ORDER[f.severity] for f in findings), default=0)
        for f in findings:
            rule_counts[f.rule] += 1
            sev_counts[f.severity] += 1
        results.append({'sequence': seq, 'worst': worst,
                        'features': baseline_features(seq),
                        'findings': [f.as_dict() for f in findings]})

    flagged = [r for r in results if r['worst'] >= floor]
    flagged.sort(key=lambda r: (-r['worst'], -len(r['findings'])))

    if args.format == 'json':
        json.dump({'source': args.path, 'sequence_field': where,
                   'n_sequences': len(results),
                   'severity_counts': dict(sev_counts),
                   'rule_counts': dict(rule_counts),
                   'results': results if not args.summary_only else []},
                  sys.stdout, indent=2)
        print()
        return 0

    if args.format == 'csv':
        w = _csv.writer(sys.stdout)
        w.writerow(['sequence', 'length', 'h_stretch', 'janin_mean', 'kd_mean',
                    'worst_severity', 'n_findings', 'rules'])
        inv = {v: k for k, v in SEV_ORDER.items()}
        for r in results:
            ft = r['features']
            w.writerow([r['sequence'], ft.get('length', 0), ft.get('h_stretch', 0),
                        ft.get('janin_mean', ''), ft.get('kd_mean', ''),
                        inv.get(r['worst'], 'INFO'), len(r['findings']),
                        '|'.join(sorted({f['rule'] for f in r['findings']}))])
        return 0

    # text
    print('=' * 74)
    print('SPPS SYNTHESIZABILITY SCAN')
    print('=' * 74)
    print('Source           : %s' % args.path)
    print('Sequence field   : %s' % where)
    print('Sequences scanned: %d' % len(results))
    print()
    print('Findings by severity:')
    for sev in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'):
        if sev_counts.get(sev):
            print('  %-9s %6d' % (sev, sev_counts[sev]))
    print()
    for cat, title in (('DATA', 'DATA INTEGRITY'),
                       ('ASSEMBLY', 'CHAIN-ASSEMBLY DIFFICULTY  (predicts crude purity)'),
                       ('PROCESS', 'PROCESS LIABILITIES  (need mitigation; not predictive)')):
        hits = [(r, c) for r, c in rule_counts.most_common() if rule_category(r) == cat]
        if not hits:
            continue
        print('%s' % title)
        for rule, cnt in hits:
            pct = 100.0 * cnt / max(1, len(results))
            lift = RULE_META.get(rule, (None, None))[1]
            ltxt = ('lift %.2f' % lift) if lift else '--'
            print('  %-26s %6d  (%5.1f%%)  %s' % (rule, cnt, pct, ltxt))
        print()
    print('Sequences at or above %s: %d of %d (%.1f%%)'
          % (args.min_severity, len(flagged), len(results),
             100.0 * len(flagged) / max(1, len(results))))

    if not args.summary_only and flagged:
        print()
        print('-' * 74)
        print('WORST %d SEQUENCES' % min(args.top, len(flagged)))
        print('-' * 74)
        for r in flagged[:args.top]:
            ft = r['features']
            print()
            print('%s   (len %d, h_stretch %d, Janin %.3f)'
                  % (r['sequence'], ft.get('length', 0), ft.get('h_stretch', 0),
                     ft.get('janin_mean', 0.0)))
            for f in r['findings']:
                if SEV_ORDER[f['severity']] >= floor or f['severity'] == 'INFO':
                    print('  [%-8s] %s' % (f['severity'], f['message']))

    print()
    print('Risk flags are mechanistic, not calibrated probabilities.')
    print('Rule sources: ../references/liability-rules.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
