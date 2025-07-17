#!/usr/bin/env python3
"""
Verification script for balanced peptide dataset
"""

import pandas as pd
import os

def verify_balanced_dataset():
    """Verify that the balanced dataset was created correctly"""
    
    # Check if balanced dataset exists
    if not os.path.exists('peptide_baza_balanced.csv'):
        print("ERROR: Balanced dataset file not found!")
        return False
    
    # Load both datasets
    original_df = pd.read_csv('peptide_baza_formatted.csv', sep=';')
    balanced_df = pd.read_csv('peptide_baza_balanced.csv', sep=';')
    
    print("Dataset Comparison:")
    print("="*50)
    print(f"Original dataset: {len(original_df)} samples")
    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(f"Additional samples: {len(balanced_df) - len(original_df)}")
    
    print("\nOriginal class distribution:")
    print(original_df['targetcol'].value_counts())
    
    print("\nBalanced class distribution:")
    print(balanced_df['targetcol'].value_counts())
    
    print("\nSample synthetic sequences:")
    synthetic_samples = balanced_df[balanced_df['id'] > original_df['id'].max()]
    print(synthetic_samples[['id', 'peptide_seq', 'peptide_len', 'synthesis_flag', 'targetcol']].head(10))
    
    return True

if __name__ == "__main__":
    verify_balanced_dataset()
