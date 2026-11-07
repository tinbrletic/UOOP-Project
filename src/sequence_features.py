"""
Sequence-derived features for peptide synthesis prediction.

Pure functions, no I/O, no external dependencies. Imported by build_dataset.py.

Two families:

  COMPOSITIONAL  counts of residues in a named group. Two conventions are provided and
                 kept separate, because they are NOT interchangeable:
                   - `paper`  the groupings used by Gutman et al. 2022, so results are
                              directly comparable to that paper's published AUC table
                   - `biochem` standard textbook biochemical groupings
                 Mixing them silently is how feature names stop meaning what they say.

  POSITIONAL     run lengths and windowed maxima. These exist because means and totals
                 discard position, and position is what drives on-resin aggregation:
                 `ILVMAAAAAA` and `AAILAAVMAA` have identical composition and identical
                 mean hydrophobicity, and behave differently during chain assembly.

The positional definitions are shared with the `spps-synthesizability-review` skill's
scanner (`.claude/skills/spps-synthesizability-review/scripts/scan_peptides.py`). They are
duplicated rather than imported so that neither depends on the other, but they must be kept
in sync — if you change a threshold or a residue set here, change it there too.
"""

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# --------------------------------------------------------------------- scales

# Kyte & Doolittle 1982, J Mol Biol 157:105
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Chou & Fasman 1978 beta-sheet propensity (P-beta). Deliberately separate from
# hydrophobicity: the two dissociate for Ala, Thr and Gln.
P_BETA = {
    'V': 1.70, 'I': 1.60, 'Y': 1.47, 'C': 1.19, 'W': 1.37, 'F': 1.38, 'L': 1.30,
    'T': 1.19, 'M': 1.05, 'A': 0.83, 'R': 0.93, 'G': 0.75, 'D': 0.54, 'K': 0.74,
    'S': 0.75, 'H': 0.87, 'N': 0.89, 'Q': 1.10, 'P': 0.55, 'E': 0.37,
}

ALIPHATIC_H = set("ILMV")     # aliphatic-hydrophobic; the paper's 'h'
BETA_BRANCHED = set("IVT")    # steric hindrance at coupling
SHEET_DISRUPTOR = set("GP")   # Pro donates no backbone NH; Gly is costly to order
PSEUDOPRO = set("STC")        # pseudoproline building blocks derive from Ser/Thr/Cys

# --------------------------------------------------------------- compositional

GROUPS_PAPER = {
    'h': ('I', 'L', 'M', 'V'),            # aliphatic-hydrophobic
    's': ('A', 'G', 'P'),                 # small
    'l': ('N', 'Q'),                      # large
    'r': ('F', 'W', 'Y'),                 # aromatic
    'p': ('C', 'S', 'T'),                 # polar
    'b': ('H', 'K', 'R'),                 # basic
    'a': ('D', 'E'),                      # acidic
}

GROUPS_BIOCHEM = {
    'tiny':      ('A', 'C', 'G', 'S', 'T'),
    'small':     ('A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V'),
    'aliphatic': ('A', 'I', 'L', 'M', 'V'),
    'aromatic':  ('F', 'W', 'Y', 'H'),
    'nonpolar':  ('A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y'),
    'polar':     ('D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T'),
    'charged':   ('D', 'E', 'H', 'K', 'R'),
    'basic':     ('H', 'K', 'R'),
    'acidic':    ('D', 'E'),
}

# The exact grouping used to generate the historical data/raw/peptide_baza.csv. Kept ONLY
# so build_dataset.py --legacy-check can prove it reproduces that file bit-for-bit. Note
# Aliphatic has no methionine: src/ingest_excel_to_db.py later added M and the CSV was
# never regenerated, so the committed script and the committed data disagree on 468 rows.
GROUPS_LEGACY = [
    ('tiny',      ('A', 'C', 'G', 'S', 'T')),
    ('small',     ('A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V')),
    ('aliphatic', ('A', 'I', 'L', 'V')),
    ('aromatic',  ('F', 'W', 'Y', 'H')),
    ('nonpolar',  ('A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W', 'Y')),
    ('polar',     ('D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T')),
    ('charged',   ('D', 'E', 'H', 'K', 'R')),
    ('basic',     ('H', 'K', 'R')),
    ('acidic',    ('D', 'E')),
]


def composition(seq, convention='paper', fractions=False):
    """Residue-group counts under one named convention.

    Returns an ordered dict of `n_<conv>_<group>` counts. With fractions=True it also
    emits `frac_<conv>_<group>`, but note these are EXACTLY n/peptide_len and peptide_len
    is itself a column, so the fractions are deterministic functions of other features.
    Including both families injects near-perfect collinearity into feature selection.
    Prefer counts; pass fractions=True only if you drop the counts or the length.
    """
    groups = GROUPS_PAPER if convention == 'paper' else GROUPS_BIOCHEM
    out = {}
    for name in sorted(groups):
        c = sum(seq.count(a) for a in groups[name])
        out['n_%s_%s' % (convention, name)] = c
        if fractions:
            out['frac_%s_%s' % (convention, name)] = c / len(seq) if seq else 0.0
    return out


# ------------------------------------------------------------------ positional

def longest_run(seq, members):
    """Length of the longest consecutive run of residues drawn from `members`."""
    best = cur = 0
    for aa in seq:
        cur = cur + 1 if aa in members else 0
        if cur > best:
            best = cur
    return best


def longest_homopolymer(seq):
    """Length of the longest run of one identical residue."""
    best = 0
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        best = max(best, j - i)
        i = j
    return best


def max_window_mean(seq, scale, w):
    """Maximum mean of `scale` over any window of width w (whole sequence if shorter)."""
    if not seq:
        return 0.0
    vals = [scale.get(a, 0.0) for a in seq]
    if len(vals) < w:
        return sum(vals) / len(vals)
    return max(sum(vals[i:i + w]) / w for i in range(len(vals) - w + 1))


def anchor_zone_fraction(seq):
    """Fraction of hydrophobic/beta-branched residues 8-25 positions from the C-terminus.

    Aggregation onset is reported in the reference literature as a window measured from
    the resin ("about 8-10 residues out"), with collapse rare beyond 25 residues. Risk is
    therefore NON-MONOTONE in length, and a translation-invariant window cannot express
    it. Peptides shorter than 9 residues have no such zone and return 0.0.
    """
    n = len(seq)
    if n < 9:
        return 0.0
    zone = seq[max(0, n - 25):n - 8]
    if not zone:
        return 0.0
    hits = sum(1 for a in zone if a in ALIPHATIC_H or a in BETA_BRANCHED)
    return hits / len(zone)


def positional(seq):
    """Run-length, windowed and terminal features. All computable before synthesis."""
    n = len(seq)
    dfree = longest_run(seq, STANDARD_AA - SHEET_DISRUPTOR)
    stretch = None
    if dfree:
        # recover the charge density of the longest disruptor-free stretch: regularity
        # only matters where solvation is not assured
        best_i = 0
        cur = cur_i = 0
        for i, aa in enumerate(seq):
            if aa not in SHEET_DISRUPTOR:
                if cur == 0:
                    cur_i = i
                cur += 1
                if cur == dfree:
                    best_i = cur_i
            else:
                cur = 0
        stretch = seq[best_i:best_i + dfree]
    charged_frac = (sum(1 for a in stretch if a in 'DEKR') / len(stretch)) if stretch else 0.0

    return {
        'h_stretch':             longest_run(seq, ALIPHATIC_H),
        'beta_branched_run':     longest_run(seq, BETA_BRANCHED),
        'homopolymer_run':       longest_homopolymer(seq),
        'no_disruptor_run':      dfree,
        'no_disruptor_charged_frac': round(charged_frac, 6),
        'max_kd_window5':        round(max_window_mean(seq, KD, 5), 6),
        'max_pbeta_window6':     round(max_window_mean(seq, P_BETA, 6), 6),
        'anchor_zone_frac':      round(anchor_zone_fraction(seq), 6),
        'n_pseudopro_sites':     sum(1 for a in seq if a in PSEUDOPRO),
    }


# --------------------------------------------------------------------- motifs

# Asp-Xaa aspartimide sensitivity, graded. Weight is ordinal, not a rate.
ASPARTIMIDE_WEIGHT = {'G': 5, 'N': 3, 'D': 2, 'S': 2, 'T': 2, 'C': 2,
                      'A': 1, 'Q': 1, 'H': 1}


def motifs(seq):
    """Named side-reaction motifs. Each is a specific, mechanistically grounded liability,
    kept as its own column rather than buried among 400 anonymous dipeptide counts."""
    n = len(seq)
    asp_score = 0
    for i in range(n - 1):
        if seq[i] == 'D':
            asp_score += ASPARTIMIDE_WEIGHT.get(seq[i + 1], 0)
    c1 = seq[-1] if n else ''
    c2 = seq[-2] if n >= 2 else ''
    return {
        'motif_DG':              seq.count('DG'),
        'aspartimide_score':     asp_score,
        'motif_nterm_Q':         1 if seq[:1] == 'Q' else 0,
        'motif_nterm_E':         1 if seq[:1] == 'E' else 0,
        'motif_cterm_P':         1 if (c1 == 'P' or c2 == 'P') else 0,
        'motif_cterm_G':         1 if (c1 == 'G' or c2 == 'G') else 0,
        'motif_cterm_NQ':        sum(1 for i, a in enumerate(seq) if a in 'NQ' and n - i <= 3),
        'n_racem_prone':         sum(1 for a in seq if a in 'HC'),
        'n_cys':                 seq.count('C'),
        'n_arg':                 seq.count('R'),
    }


def all_features(seq, convention='paper', fractions=False):
    """Every sequence-derived feature, as one flat dict."""
    seq = (seq or '').strip().upper()
    out = {'peptide_len': len(seq)}
    out.update(composition(seq, convention, fractions))
    out.update(positional(seq))
    out.update(motifs(seq))
    return out
