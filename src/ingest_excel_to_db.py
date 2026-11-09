#!/usr/bin/env python3
"""
DEPRECATED. Superseded by src/build_dataset.py.

This script loaded the source spreadsheet into a local MySQL database, from which the
CSVs were exported by hand. It has been retired for three reasons:

1. It required a database, and carried its credential in the source file. The credential
   that was committed here must be treated as compromised and rotated; removing it from
   this file does NOT remove it from git history.

2. It did not reproduce the data it was supposed to have produced. Its Aliphatic group
   was changed to include methionine (commit 388d6b4) and data/raw/peptide_baza.csv was
   never regenerated, so the script and the data disagreed on the 468 rows containing M.

3. Its compositional() returned a [count, fraction] pair per group -- 18 values for 9
   groups -- and the INSERT wrote only the first nine, into nine differently-named
   columns. Eight of the nine held a different quantity than their name said, and the
   polar, charged, basic and acidic counts were absent from the dataset altogether.

Use instead:

    python src/build_dataset.py                 # build data/processed/peptide_features.csv
    python src/build_dataset.py --legacy-check  # prove the rebuild reproduces the old CSV

build_dataset.py reads the spreadsheet directly, needs no database and no credential, and
names every column for what it contains. Its --legacy-check reproduces all 46,046 values
of the historical dataset exactly, including both faults above, which is the evidence that
nothing was lost in the replacement.
"""

import sys

sys.exit(__doc__)
