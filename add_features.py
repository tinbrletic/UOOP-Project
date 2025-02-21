import pandas as pd
from collections import Counter
import csv

# Define amino acids (alphabetical order)
AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

def calculate_x4(seq):
    """One-hot encode N-terminal amino acid"""
    if pd.isna(seq) or len(str(seq)) == 0:
        return [0]*len(AMINO_ACIDS)
    first_aa = str(seq)[0]
    return [1 if aa == first_aa else 0 for aa in AMINO_ACIDS]

def calculate_x5(seq):
    """Count amino acid occurrences"""
    if pd.isna(seq) or len(str(seq)) == 0:
        return [0]*len(AMINO_ACIDS)
    counter = Counter(str(seq))
    return [counter.get(aa, 0) for aa in AMINO_ACIDS]

def calculate_x8(seq):
    """Count dipeptide bonds"""
    if pd.isna(seq) or len(str(seq)) < 2:
        return [0]*(len(AMINO_ACIDS)**2)
    pairs = [a+b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    sequence = str(seq)
    pair_counts = Counter(sequence[i:i+2] for i in range(len(sequence)-1))
    return [pair_counts.get(pair, 0) for pair in pairs]

# Load dataset with proper column names
df = pd.read_csv('peptide_baza.csv', 
                 sep=';', 
                 header=0,
                 names=["id", "peptide_seq", "peptide_len", "synthesis_flag",
                        "hydrophobic_janin", "hydrophobic_engleman", 
                        "hydrophobic_moment", "aliphatic_index",
                        "isoelectric_point", "charge", "tiny_group",
                        "small_group", "aliphatic_group", "aromatic_group",
                        "non-polar_group", "polar_group", "charged_group",
                        "basic_group", "acidic_group", "cruciani_prp1",
                        "cruciani_prp2", "cruciani_prp3", "instability_index",
                        "boman", "hydrophobic_kyte-doolittle",
                        "hydrophobic_hopp-woods", "hydrophobic_cornette",
                        "hydrophobic_eisenberg", "hydrophobic_roseman",
                        "targetcol"])

print("Columns in DataFrame:", df.columns.tolist())
print("First 3 rows:\n", df.head(3))

# Generate X4 features
x4_cols = [f'X4_{aa}' for aa in AMINO_ACIDS]
df[x4_cols] = df['peptide_seq'].apply(
    lambda x: pd.Series(calculate_x4(x))
)

# Generate X5 features
x5_cols = [f'X5_{aa}' for aa in AMINO_ACIDS]
df[x5_cols] = df['peptide_seq'].apply(
    lambda x: pd.Series(calculate_x5(x))
)

# Generate X8 features
x8_cols = [f'X8_{a+b}' for a in AMINO_ACIDS for b in AMINO_ACIDS]
df[x8_cols] = df['peptide_seq'].apply(
    lambda x: pd.Series(calculate_x8(x))
)

# Save enhanced dataset
# df.to_csv('peptide_baza_with_features.csv', sep=";", index=False);
df.to_csv('peptide_baza_formatted.csv', 
          sep=';', 
          quoting=csv.QUOTE_NONNUMERIC,
          quotechar='"',
          index=False)
