#!/usr/bin/env python3
"""
Scan peptide sequences for Fmoc-SPPS synthesis liabilities.

Deterministic rule engine. Every rule is sequence-computable and carries a source.
Reads CSV/TSV/XLSX/FASTA/plain-text, finds the sequence column, emits findings.

Usage:
    python scan_peptides.py <file> [--column NAME] [--sep ';'] [--format text|json|csv]
                                   [--top N] [--min-severity LOW] [--summary-only]

Exit codes: 0 = scanned, 2 = could not find sequences.

Rule sources are listed in ../references/factor-catalogue.md. This reports MECHANISTIC
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

ALIPHATIC_H = set("ILMV")      # the classic 'h' set: aliphatic-hydrophobic
BETA_BRANCHED = set("IVT")     # slow coupling, steric
PSEUDOPRO_PRECURSOR = set("STC")  # pseudoproline building blocks derive from Ser/Thr/Cys

# Gly and Pro break beta-sheet: Pro cannot donate a backbone NH and is conformationally
# restricted; Gly is flexible and entropically costly to order. A long stretch containing
# NEITHER is free to form regular extended structure. This is why poly-Ala aggregates --
# the canonical difficult sequence is decaalanine, which the ILMV run rule scores 0.
# Ala's own beta-sheet propensity is LOW (Chou-Fasman P-beta 0.83); the driver is backbone
# regularity with no disruptor and no steric relief, not side-chain hydrophobicity.
SHEET_DISRUPTOR = set("GP")

# Chou & Fasman 1978 beta-sheet propensity (P-beta). Used for a propensity-weighted window
# that is independent of hydrophobicity, since the two dissociate (Ala, Thr, Gln).
P_BETA = {
    'V': 1.70, 'I': 1.60, 'Y': 1.47, 'C': 1.19, 'W': 1.37, 'F': 1.38, 'L': 1.30,
    'T': 1.19, 'M': 1.05, 'A': 0.83, 'R': 0.93, 'G': 0.75, 'D': 0.54, 'K': 0.74,
    'S': 0.75, 'H': 0.87, 'N': 0.89, 'Q': 1.10, 'P': 0.55, 'E': 0.37,
}

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
#   ASSEMBLY - bears on whether the chain assembles cleanly (solvation/aggregation and
#              steric access). Mechanistically the dominant cause of incomplete coupling.
#   PROCESS  - a discrete side reaction with a known mitigation. Whether it shows up in a
#              given outcome measure depends entirely on what that measure detects.
#
# NO LIFT VALUES ARE HARDCODED, deliberately. Predictive power is a property of a dataset
# and its outcome definition, not of the chemistry. Derive it for your own data with
# --calibrate; see "Calibration" in SKILL.md.
RULE_CATEGORY = {
    'agg.hydrophobic_window':      'ASSEMBLY',
    'agg.h_stretch':               'ASSEMBLY',
    'agg.beta_branched_run':       'ASSEMBLY',
    'agg.homopolymer_run':         'ASSEMBLY',
    'agg.no_disruptor_window':     'ASSEMBLY',
    'agg.sheet_propensity_window': 'ASSEMBLY',
    'agg.anchor_zone':             'ASSEMBLY',
    'agg.pseudoproline_site':      'ASSEMBLY',
    # Length is about CUMULATIVE YIELD, not aggregation. Aggregation risk is reported in
    # the literature as non-monotone in length (a window measured from the resin), so the
    # two are deliberately separate rules; see agg.anchor_zone.
    'agg.length':                  'YIELD',
    'side.diketopiperazine':  'PROCESS',
    'side.aspartimide':       'PROCESS',
    'side.pyroglutamate':     'PROCESS',
    'side.oxidation':         'PROCESS',
    'side.racemization':      'PROCESS',
    'side.arg_lactam':        'PROCESS',
    'side.disulfide':         'PROCESS',
    'stability.deamidation':  'PROCESS',
    'stability.deamidation_cterm': 'PROCESS',
    'seq.nonstandard':        'DATA',
    'seq.empty':              'DATA',
}

# Sequences the literature names as known-difficult, with what makes each one hard.
# Used by --selftest: when no labelled outcomes exist, the nearest available validity
# check is whether the rules fire on cases a source already calls failures.
SELFTEST = [
    ('AAAAAAAAAA', 'decaalanine - the canonical difficult sequence; beta-structure and '
                   'severe resin shrinkage, NMR line-broadening 8-10 residues from resin'),
    ('VQAAIDYING', 'ACP(65-74) - the standard difficult-coupling benchmark'),
    ('VQLLLLIDYING', 'contrived: long aliphatic run inside the collapse zone'),
    ('HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR', 'a 30-mer with a C-terminal hydrophobic stretch'),
    ('QLLLLVDGKP', 'contrived: N-terminal Gln, Asp-Gly, C-terminal Pro, hydrophobic run'),
    ('AGSGAGSGAK', 'benign control - Gly/Ser rich, should stay quiet'),
    ('EEKKEEKKEEKK', 'benign control - charged, highly soluble, should stay quiet'),
]


def rule_category(rule):
    return RULE_CATEGORY.get(rule, 'PROCESS')


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


def longest_homopolymer(seq):
    """Return (length, start_index_1based) of the longest run of one identical residue."""
    best = best_start = 0
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        if j - i > best:
            best, best_start = j - i, i
        i = j
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
                         'Length %d. Cumulative stepwise yield loss grows with length.' % n))

    # --- backbone regularity: catches poly-Ala, which the ILMV run rule cannot see ----
    homo, hpos = longest_homopolymer(seq)
    if homo >= 5:
        f.append(Finding('agg.homopolymer_run', 'CRITICAL' if homo >= 8 else 'HIGH',
                         'Run of %d identical residues (%s) at position %d. Maximal backbone '
                         'regularity with no disruption; homopolymer stretches of small '
                         'residues aggregate even when not hydrophobic.'
                         % (homo, seq[hpos - 1], hpos), [hpos]))

    # Regularity alone is not enough: aggregation needs regular backbone AND poor
    # solvation. A disruptor-free stretch of charged residues (e.g. EEKKEEKKEEKK) is
    # highly soluble and does not aggregate, so gate on charge density.
    dfree, dpos = longest_run(seq, STANDARD_AA - SHEET_DISRUPTOR)
    if dfree >= 10:
        stretch = seq[dpos - 1:dpos - 1 + dfree]
        charged_frac = sum(1 for a in stretch if a in 'DEKR') / len(stretch)
        if charged_frac < 0.40:
            f.append(Finding('agg.no_disruptor_window', 'HIGH' if dfree >= 14 else 'MEDIUM',
                             'Stretch of %d residues from position %d with no Gly or Pro and '
                             'only %.0f%% charged. Neither beta-sheet disruptor is present and '
                             'solvation is not assured, so the backbone is free to adopt '
                             'regular extended structure.'
                             % (dfree, dpos, 100 * charged_frac), [dpos]))

    pw = window_means(seq, P_BETA, 6)
    if pw:
        pstart, pmax = max(pw, key=lambda t: t[1])
        if pmax >= 1.30:
            f.append(Finding('agg.sheet_propensity_window', 'MEDIUM',
                             'High beta-sheet propensity over 6 residues (mean P-beta %.2f) at '
                             'position %d: %s. Independent of hydrophobicity - the two '
                             'dissociate for Ala, Thr and Gln.'
                             % (pmax, pstart, seq[pstart - 1:pstart + 5]), [pstart]))

    # --- anchored position: risk is reported as a WINDOW measured from the resin --------
    # Houben-Weyl E22: "the critical region was the zone about 8-10 residues out from the
    # resin"; "the synthesis will rarely collapse after 25 residues". Aggregation risk is
    # NON-MONOTONE in length, so this is indexed from the C-terminal anchor, not global.
    if n >= 8:
        lo_i, hi_i = max(0, n - 25), n - 8   # residues 8..25 out from the C-terminal anchor
        zone = seq[lo_i:hi_i]
        if zone:
            zh = sum(1 for a in zone if a in ALIPHATIC_H or a in BETA_BRANCHED)
            frac = zh / len(zone)
            zd = sum(1 for a in zone if a in SHEET_DISRUPTOR)
            if frac >= 0.40 and zd == 0:
                f.append(Finding('agg.anchor_zone', 'HIGH',
                                 'Collapse zone (residues 8-%d from the C-terminal anchor, '
                                 '"%s") is %.0f%% hydrophobic/beta-branched with no Gly or '
                                 'Pro. This is the window the literature reports aggregation '
                                 'onset in; position is measured from the resin, not globally.'
                                 % (min(25, n), zone, 100 * frac), [lo_i + 1]))
            elif frac >= 0.40:
                f.append(Finding('agg.anchor_zone', 'MEDIUM',
                                 'Collapse zone (residues 8-%d from the C-terminal anchor) is '
                                 '%.0f%% hydrophobic/beta-branched.'
                                 % (min(25, n), 100 * frac), [lo_i + 1]))

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
                         'give des-Arg deletions; recoupling is commonly required. CAUTION: if '
                         'the outcome measure derives from positive-mode MS, Arg content '
                         'inflates apparent purity (basic residues ionise well), so this rule '
                         'can correlate with SUCCESS. Calibrate before using it as a signal.'
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
                         'matters. Fires on a large fraction of typical sets, so it is a '
                         'protocol note rather than a discriminating signal.'
                         % ', '.join('%s%d' % (a, p) for p, a in ox)))

    # Deamidation. Both a motif effect and a POSITIONAL effect: the same Gln has been
    # reported to deamidate at 3 residues from the C-terminus and not at 3 from the
    # N-terminus, so proximity to the C-terminus is a signal in its own right.
    for m in re.finditer('N(?=[GS])', seq):
        f.append(Finding('stability.deamidation', 'LOW',
                         'Asn-%s at position %d. Classic deamidation hotspot.'
                         % (seq[m.start() + 1], m.start() + 1), [m.start() + 1]))
    tail = [(i + 1, a) for i, a in enumerate(seq) if a in 'NQ' and n - i <= 3]
    if tail:
        f.append(Finding('stability.deamidation_cterm', 'MEDIUM',
                         'Asn/Gln within 3 residues of the C-terminus (%s). Deamidation is '
                         'position-dependent, not only motif-dependent: the same residue can '
                         'deamidate near the C-terminus and not near the N-terminus. A '
                         'C-terminal Asx-alpha-amide is especially susceptible.'
                         % ', '.join('%s%d' % (a, p) for p, a in tail),
                         [p for p, _ in tail]))
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


# ---------------------------------------------------------------- calibration

def auc_rank(scores, labels):
    """Rank-based AUC for predicting label==1. Ties get average ranks."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            rk[k] = avg
        i = j
    pos = sum(1 for _, l in pairs if l == 1)
    neg = n - pos
    if pos < 3 or neg < 3:
        return None
    s = sum(rk[k] for k in range(n) if pairs[k][1] == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


CONT_FEATURES = ('length', 'janin_mean', 'kd_mean', 'h_stretch', 'max_kd_window')


def _cont(seq):
    ft = baseline_features(seq)
    w = window_means(seq, KD, 5)
    ft['max_kd_window'] = round(max(v for _, v in w), 4) if w else 0.0
    return ft


def calibrate(pairs, label_name, failure_value):
    """pairs: list of (sequence, raw_label). Derive rule lift and feature AUC on THIS data."""
    data = []
    for seq, raw in pairs:
        seq = (seq or '').strip().upper()
        if not seq:
            continue
        fail = 1 if str(raw).strip() == str(failure_value) else 0
        _, findings = scan_sequence(seq)
        data.append((seq, fail, {f.rule for f in findings}, _cont(seq)))

    n = len(data)
    nfail = sum(d[1] for d in data)
    print('=' * 74)
    print('CALIBRATION ON THIS DATASET')
    print('=' * 74)
    print('n = %d   outcome column = %r   value counted as failure = %r'
          % (n, label_name, failure_value))
    if not nfail or nfail == n:
        print('\nAll rows have the same outcome. Cannot calibrate.')
        print('Check --failure-value against the actual column values.')
        return
    print('failures = %d (%.1f%%)   successes = %d' % (nfail, 100.0 * nfail / n, n - nfail))
    print()
    print('Lift = P(failure | rule fires) / P(failure | it does not). 1.0 = no signal.')
    print('Below 1.0 means the rule fires MORE often on successes - investigate why.')
    print()
    base = nfail / n
    print('%-26s %5s %9s %9s %7s %s' %
          ('RULE', 'fires', 'fail|fire', 'fail|not', 'lift', 'power'))
    print('-' * 72)
    rows = []
    skipped = []
    for rule in sorted(RULE_CATEGORY):
        fires = [d for d in data if rule in d[2]]
        rest = [d for d in data if rule not in d[2]]
        if not fires or not rest:
            skipped.append((rule, len(fires), 'never fires' if not fires else 'always fires'))
            continue
        a = sum(d[1] for d in fires) / len(fires)
        b = sum(d[1] for d in rest) / len(rest)
        # Expected failures among the firing rows under the null. Below ~10 the lift is
        # dominated by counting noise and the rule is UNTESTED on this data, not refuted.
        exp_fail = len(fires) * base
        rows.append((a / b if b else float('inf'), rule, len(fires), a, b, exp_fail))
    for lift, rule, nf, a, b, exp in sorted(rows, reverse=True):
        if exp < 5:
            tag = 'UNTESTED (exp. %.1f failures)' % exp
        elif exp < 10:
            tag = 'weak (exp. %.1f failures)' % exp
        else:
            tag = ''
        print('%-26s %5d %9.3f %9.3f %7.2f %s' % (rule, nf, a, b, lift, tag))
    for rule, nf, why in skipped:
        print('%-26s %5d %9s %9s %7s %s' % (rule, nf, '-', '-', '-', why))
    print()
    print('power: expected failures among the firing rows under the null (fires x base rate).')
    print('A rule marked UNTESTED could not have shown an effect on this data at any')
    print('strength. That is a statement about the dataset, not about the chemistry --')
    print('do not read it as the rule being disproved.')

    labels = [d[1] for d in data]
    print()
    print('%-16s %9s %8s %7s %s' % ('FEATURE', 'AUC(fail)', 'distinct', 'zeros', 'note'))
    print('-' * 62)
    for f in CONT_FEATURES:
        vals = [d[3].get(f, 0) for d in data]
        a = auc_rank(vals, labels)
        distinct = len(set(vals))
        zfrac = sum(1 for v in vals if v == 0) / len(vals)
        # A feature that is constant, near-constant, or mostly zero cannot carry signal on
        # this data whatever its mechanism. Anchored-window features are the usual
        # casualty: a window measured 8-25 residues from one terminus is only a few
        # residues wide on short peptides, and empty on the shortest.
        note = ''
        if distinct <= 2:
            note = 'DEGENERATE (near-constant here)'
        elif zfrac >= 0.30:
            note = 'DEGENERATE (%.0f%% zero)' % (100 * zfrac)
        elif distinct < 8:
            note = 'coarse (%d levels)' % distinct
        print('%-16s %9s %8d %6.0f%% %s'
              % (f, '%.4f' % a if a is not None else 'n/a', distinct, 100 * zfrac, note))
    print()
    print('A DEGENERATE feature has too little variation on this data to show an effect.')
    print('Check the definition against the data range before concluding anything: an')
    print('anchored window needs peptides long enough for the window to exist.')

    # --- cohort-confound check -------------------------------------------------
    by_len = {}
    for seq, fail, _, _ in data:
        c = by_len.setdefault(len(seq), [0, 0])
        c[0] += 1
        c[1] += fail
    worst = max(by_len.items(), key=lambda kv: kv[1][1])
    L, (ln, lf) = worst
    share_fail = lf / nfail
    share_rows = ln / n
    print()
    print('COHORT CHECK')
    print('-' * 27)
    print('%-16s %6s %6s %9s' % ('length', 'n', 'fail', 'fail rate'))
    for k in sorted(by_len):
        c = by_len[k]
        print('%-16d %6d %6d %9.3f' % (k, c[0], c[1], c[1] / c[0]))
    # Two different things make length do structural work, and they need separate tests.
    #   (a) ONE length value dominates both rows and failures -> a distinct sub-library.
    #   (b) failure rate climbs MONOTONICALLY across lengths -> no single value dominates,
    #       but length still carries most of the signal. A check written only for (a)
    #       stays silent on a relationship running from 5% to 76%.
    dominant = share_fail >= 0.40 and share_rows >= 0.20

    # (b): split into length quartiles by row count and compare failure rates.
    ordered = sorted(data, key=lambda d: len(d[0]))
    q = max(1, len(ordered) // 4)
    bands = [ordered[0:q], ordered[q:2 * q], ordered[2 * q:3 * q], ordered[3 * q:]]
    bands = [b for b in bands if len(b) >= 10]
    rates = [(len(b[0][0]), len(b[-1][0]), sum(d[1] for d in b) / len(b), len(b))
             for b in bands]
    gradient = False
    if len(rates) >= 3:
        lo = min(r[2] for r in rates)
        hi = max(r[2] for r in rates)
        gradient = hi >= 0.10 and (lo == 0.0 or hi / lo >= 2.5)

    print()
    if dominant:
        print('WARNING: length %d holds %.0f%% of all failures and %.0f%% of all rows.'
              % (L, 100 * share_fail, 100 * share_rows))
        print('That is a distinct sub-library, not a chain-length effect.')
    if gradient:
        print('WARNING: failure rate varies strongly across length bands:')
        for a, b, rate, cnt in rates:
            print('    len %2d-%-2d  n=%-5d  fail rate %.3f' % (a, b, cnt, rate))
        print('Length carries much of the signal even though no single value dominates.')

    if dominant or gradient:
        # Stratify on whichever structure fired: the dominant value, else the largest band.
        if dominant:
            sub = [d for d in data if len(d[0]) == L]
            label = 'length == %d' % L
        else:
            sub = max(bands, key=len)
            label = 'len %d-%d' % (len(sub[0][0]), len(sub[-1][0]))
        sl = [d[1] for d in sub]
        print()
        print('Any feature correlated with length may be acting as a cohort indicator')
        print('rather than a chemical signal. Re-checking within %s (n=%d):'
              % (label, len(sub)))
        print()
        print('  %-16s %9s %9s' % ('FEATURE', 'AUC all', 'AUC in-stratum'))
        print('  ' + '-' * 38)
        for f in CONT_FEATURES:
            a_all = auc_rank([d[3].get(f, 0) for d in data], labels)
            a_sub = auc_rank([d[3].get(f, 0) for d in sub], sl)
            print('  %-16s %9s %9s'
                  % (f, '%.4f' % a_all if a_all is not None else 'n/a',
                     '%.4f' % a_sub if a_sub is not None else 'n/a'))
        print()
        print('  A feature whose AUC collapses in-stratum was riding the cohort.')
        print('  A feature that holds up is measuring something real.')
    else:
        print('No single length dominates the failures, and failure rate does not')
        print('vary strongly across length bands. No cohort warning.')


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


def load_pairs(path, seq_column, label_column, sep=None):
    """Return [(sequence, raw_label)] from a delimited file or spreadsheet."""
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.xlsx', '.xlsm', '.xltx'):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            for hdr_i, hdr in enumerate(rows[:5]):
                cells = [str(c).strip() if c is not None else '' for c in hdr]
                if label_column in cells and (seq_column is None or seq_column in cells):
                    li = cells.index(label_column)
                    si = cells.index(seq_column) if seq_column else next(
                        (i for i, c in enumerate(cells) if c.lower() in SEQ_COL_HINTS), 0)
                    for r in rows[hdr_i + 1:]:
                        if si < len(r) and li < len(r) and r[si] and r[li] is not None:
                            v = str(r[si]).strip()
                            if _is_sequence(v):
                                out.append((v, r[li]))
                    wb.close()
                    return out
        wb.close()
        return []

    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
        sample = fh.read(8192)
        fh.seek(0)
        if sep is None:
            try:
                sep = _csv.Sniffer().sniff(sample, delimiters=';,\t|').delimiter
            except Exception:
                sep = ';' if sample.count(';') > sample.count(',') else ','
        rows = list(_csv.reader(fh, delimiter=sep))
    if not rows:
        return []
    header = [h.strip().strip('"') for h in rows[0]]
    if label_column not in header:
        sys.exit('Label column %r not found. Available: %s' % (label_column, header))
    li = header.index(label_column)
    if seq_column:
        if seq_column not in header:
            sys.exit('Sequence column %r not found. Available: %s' % (seq_column, header))
        si = header.index(seq_column)
    else:
        si = next((header.index(h) for hint in SEQ_COL_HINTS for h in header
                   if h.lower() == hint), None)
        if si is None:
            scores = sorted(((_looks_like_sequences([r[i] for r in rows[1:400] if i < len(r)]), i)
                             for i in range(len(header))), reverse=True)
            if not scores or scores[0][0] < 0.3:
                return []
            si = scores[0][1]
    return [(r[si].strip().strip('"'), r[li].strip().strip('"'))
            for r in rows[1:] if si < len(r) and li < len(r) and r[si].strip()]


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?')
    ap.add_argument('--column', help='sequence column name (else auto-detected)')
    ap.add_argument('--sep', help='delimiter for text files (else sniffed)')
    ap.add_argument('--format', choices=('text', 'json', 'csv'), default='text')
    ap.add_argument('--top', type=int, default=15, help='worst N sequences to list')
    ap.add_argument('--min-severity', default='MEDIUM',
                    choices=tuple(SEV_ORDER), help='floor for per-sequence listing')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--label-column',
                    help='outcome column; given, the script calibrates rules on THIS dataset '
                         'instead of reporting borrowed numbers')
    ap.add_argument('--failure-value',
                    help='value in --label-column that means "failed/problematic". '
                         'Default: the minority value, reported explicitly.')
    ap.add_argument('--selftest', action='store_true',
                    help='run the rules against sequences the literature names as '
                         'known-difficult, plus benign controls. Use when no labelled '
                         'outcomes exist - it is the nearest available validity check.')
    args = ap.parse_args()

    if args.selftest:
        print('RULE SELF-TEST')
        print('=' * 74)
        print('Known-difficult sequences from the literature, and benign controls.')
        print('A difficult case that draws nothing, or a control that lights up, is a '
              'rule defect.\n')
        for seq, why in SELFTEST:
            _, findings = scan_sequence(seq)
            worst = max((SEV_ORDER[x.severity] for x in findings), default=0)
            inv = {v: k for k, v in SEV_ORDER.items()}
            print('%-32s %-9s %s' % (seq, inv[worst], why))
            for x in sorted(findings, key=lambda y: -SEV_ORDER[y.severity]):
                if SEV_ORDER[x.severity] >= SEV_ORDER['MEDIUM']:
                    print('    [%-8s] %s' % (x.severity, x.rule))
            if not findings:
                print('    (nothing)')
            print()
        return 0

    if not args.path:
        sys.exit('A file is required unless --selftest is given.')
    if not os.path.exists(args.path):
        sys.exit('No such file: %s' % args.path)

    if args.label_column:
        pairs = load_pairs(args.path, args.column, args.label_column, args.sep)
        if not pairs:
            print('Could not read sequence+label pairs from %s' % args.path, file=sys.stderr)
            return 2
        fv = args.failure_value
        if fv is None:
            counts = Counter(str(l).strip() for _, l in pairs)
            fv = min(counts, key=counts.get)
            print('--failure-value not given; using minority value %r %s\n'
                  % (fv, dict(counts)))
        calibrate(pairs, args.label_column, fv)
        return 0

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
                       ('ASSEMBLY', 'CHAIN-ASSEMBLY DIFFICULTY  (solvation / steric access)'),
                       ('YIELD', 'CUMULATIVE YIELD  (length; distinct from aggregation)'),
                       ('PROCESS', 'PROCESS LIABILITIES  (discrete, each with a mitigation)')):
        hits = [(r, c) for r, c in rule_counts.most_common() if rule_category(r) == cat]
        if not hits:
            continue
        print('%s' % title)
        for rule, cnt in hits:
            pct = 100.0 * cnt / max(1, len(results))
            print('  %-26s %6d  (%5.1f%% of sequences)' % (rule, cnt, pct))
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
    print('Rule sources: ../references/factor-catalogue.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
