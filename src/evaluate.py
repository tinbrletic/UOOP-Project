#!/usr/bin/env python3
"""
Honest evaluation of the rebuilt feature table.

Answers four questions the existing sweep does not answer directly:

  1. Does anything beat the published two-feature baseline (length + Janin, AUC 0.773)?
  2. Does the signal survive stratification, or is it riding the length-15 cohort?
  3. Do the 400 X8 dipeptide features add anything over the named positional features?
  4. What is the failure-class recall -- the number a chemist actually asks for?

Design choices, all deliberate:

  * FAILURE IS THE POSITIVE CLASS. "Which of my peptides will give me trouble?" is a
    question about the minority class. Reporting accuracy or positive-class F1 on an 87/13
    split rewards a constant classifier: it scores 0.869 accuracy and 0.930 F1 with MCC
    0.000. Those metrics are excluded here on purpose.
  * Everything that learns from data sits INSIDE the cross-validation pipeline -- scaling,
    resampling, the model. Nothing is fitted on the full dataset.
  * Class imbalance is handled with class_weight='balanced' by default rather than
    synthetic oversampling. At n=1771 with 232 failures, reweighting is better behaved than
    interpolating new points in a space containing one-hot and count features. --smote
    switches to SMOTE inside the pipeline for comparison.
  * Out-of-fold predictions are pooled before scoring, so each peptide is scored exactly
    once per repeat.

Usage:
    python src/evaluate.py                      # all feature sets, both strata
    python src/evaluate.py --smote              # SMOTE in-pipeline instead of reweighting
    python src/evaluate.py --repeats 10         # more repeats (default 5)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sequence_features as sf  # noqa: E402

DATA = os.path.join('data', 'processed', 'peptide_features.csv')
AA = list('ACDEFGHIKLMNPQRSTVWY')


def add_x_blocks(df):
    """Recreate the historical X4 (N-terminal one-hot), X5 (residue counts) and
    X8 (400 dipeptide counts) blocks, so they can be compared against the named set."""
    seqs = df['peptide_seq'].tolist()
    out = {}
    for a in AA:
        out['X4_' + a] = [1 if s[:1] == a else 0 for s in seqs]
        out['X5_' + a] = [s.count(a) for s in seqs]
    for a in AA:
        for b in AA:
            pair = a + b
            out['X8_' + pair] = [sum(1 for i in range(len(s) - 1) if s[i:i + 2] == pair)
                                 for s in seqs]
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def feature_sets(df, base_cols):
    """named -> the rebuilt table only. baseline -> the published two-feature pair.

    base_cols must be the columns BEFORE add_x_blocks, or the named set silently
    swallows the X blocks and the comparison becomes meaningless.
    """
    drop = ['id', 'peptide_seq', 'targetcol']
    named = [c for c in base_cols if c not in drop]
    return {
        'baseline (len+Janin)': ['peptide_len', 'hydrophobic_janin'],
        'named (rebuilt)':      named,
        'named + X4/X5':        named + [c for c in df.columns
                                         if c.startswith(('X4_', 'X5_'))],
        'named + X4/X5/X8':     named + [c for c in df.columns
                                         if c.startswith(('X4_', 'X5_', 'X8_'))],
    }


def models(smote):
    base = {
        'LogReg': LogisticRegression(max_iter=5000, class_weight='balanced'),
        'RandForest': RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                             random_state=42, n_jobs=-1),
    }
    out = {}
    for name, clf in base.items():
        steps = [('scale', StandardScaler())]
        if smote:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            steps.append(('smote', SMOTE(random_state=42, k_neighbors=5)))
            steps.append(('clf', clf))
            out[name] = ImbPipeline(steps)
        else:
            steps.append(('clf', clf))
            out[name] = Pipeline(steps)
    return out


def evaluate(X, y, model, repeats, seed=42):
    """Pooled out-of-fold scores, one row per peptide per repeat."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    n_rep = repeats
    per_repeat = {m: [] for m in ('MCC', 'AUC', 'PR-AUC', 'Recall_fail', 'Prec_fail')}
    fold_i = 0
    oof_p = np.zeros(len(y))
    oof_h = np.zeros(len(y))
    for train, test in cv.split(X, y):
        m = model
        m.fit(X.iloc[train], y[train])
        oof_p[test] = m.predict_proba(X.iloc[test])[:, 1]
        oof_h[test] = m.predict(X.iloc[test])
        fold_i += 1
        if fold_i % 5 == 0:                     # one full pass over the data
            per_repeat['MCC'].append(matthews_corrcoef(y, oof_h))
            per_repeat['AUC'].append(roc_auc_score(y, oof_p))
            per_repeat['PR-AUC'].append(average_precision_score(y, oof_p))
            per_repeat['Recall_fail'].append(recall_score(y, oof_h, zero_division=0))
            per_repeat['Prec_fail'].append(precision_score(y, oof_h, zero_division=0))
            oof_p = np.zeros(len(y))
            oof_h = np.zeros(len(y))
    assert len(per_repeat['MCC']) == n_rep
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in per_repeat.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--smote', action='store_true',
                    help='SMOTE inside the pipeline instead of class_weight=balanced')
    ap.add_argument('--data', default=DATA)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit('Missing %s -- run: python src/build_dataset.py' % args.data)
    df = pd.read_csv(args.data, sep=';')
    base_cols = list(df.columns)
    df = add_x_blocks(df)

    # FAILURE is the positive class.
    y_all = (df['targetcol'] == 0).astype(int).values

    strata = {
        'ALL (n=%d)' % len(df): np.ones(len(df), dtype=bool),
        'LEN==15 (n=%d)' % int((df.peptide_len == 15).sum()): (df.peptide_len == 15).values,
    }
    fsets = feature_sets(df, base_cols)
    mdls = models(args.smote)

    print('=' * 86)
    print('EVALUATION -- failure is the positive class')
    print('=' * 86)
    print('balancing : %s' % ('SMOTE in-pipeline' if args.smote else "class_weight='balanced'"))
    print('CV        : 5-fold stratified x %d repeats, everything fitted inside the fold'
          % args.repeats)
    print('reference : the source paper reports AUC 0.773 for length + a hydrophobicity scale')
    print()

    for sname, mask in strata.items():
        sub = df[mask]
        y = y_all[mask]
        print('-' * 86)
        print('%s   failures %d (%.1f%%)' % (sname, y.sum(), 100 * y.mean()))
        print('-' * 86)
        print('%-22s %-11s %12s %12s %12s %12s' %
              ('FEATURES', 'MODEL', 'MCC', 'AUC', 'PR-AUC', 'Recall_fail'))
        for fname, cols in fsets.items():
            cols = [c for c in cols if c in sub.columns]
            X = sub[cols].astype(float)
            if X.shape[1] == 0:
                continue
            for mname, model in mdls.items():
                r = evaluate(X, y, model, args.repeats)
                print('%-22s %-11s %6.3f+-%-5.3f %6.3f+-%-5.3f %6.3f+-%-5.3f %6.3f+-%-5.3f'
                      % (fname if mname == 'LogReg' else '', mname,
                         r['MCC'][0], r['MCC'][1], r['AUC'][0], r['AUC'][1],
                         r['PR-AUC'][0], r['PR-AUC'][1],
                         r['Recall_fail'][0], r['Recall_fail'][1]))
        print()

    print('Reading guide:')
    print('  * A feature set that does not beat "baseline (len+Janin)" has not earned its')
    print('    complexity.')
    print('  * Compare ALL against LEN==15. A model whose AUC drops sharply in the')
    print('    homogeneous stratum was partly predicting which sub-library a peptide')
    print('    came from, not whether it would synthesise.')
    print('  * PR-AUC is the honest headline at this class balance; chance equals the')
    print('    failure rate, not 0.5.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
