#!/usr/bin/env python3
"""
SMOTE-based Peptide Dataset Balancing Script

This script creates a balanced version of the peptide dataset by:
1. Loading the original dataset
2. Applying SMOTE to generate synthetic samples for the minority class
3. Creating synthetic peptide sequences for the new samples
4. Saving the balanced dataset

Usage: python smote_balance.py
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import random
import sys

def create_balanced_peptide_dataset():
    """
    Create a balanced peptide dataset using SMOTE
    """
    try:
        print("Loading peptide dataset...")
        df = pd.read_csv('peptide_baza_formatted.csv', sep=';')
        
        print(f"Original dataset: {len(df)} samples")
        print("Original class distribution:")
        print(f"  synthesis_flag=True (targetcol=1): {sum(df['targetcol'] == 1)}")
        print(f"  synthesis_flag=False (targetcol=0): {sum(df['targetcol'] == 0)}")
        
        # Prepare features for SMOTE
        exclude_cols = ['id', 'peptide_seq', 'synthesis_flag', 'targetcol']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].fillna(0)
        y = df['targetcol']
        
        print(f"Using {len(feature_columns)} features for SMOTE")
        
        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE: {len(X_resampled)} samples")
        print("New class distribution:")
        print(f"  synthesis_flag=True (targetcol=1): {sum(y_resampled == 1)}")
        print(f"  synthesis_flag=False (targetcol=0): {sum(y_resampled == 0)}")
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(X_resampled, columns=feature_columns)
        balanced_df['targetcol'] = y_resampled
        balanced_df['synthesis_flag'] = (y_resampled == 1)
        
        # Add IDs
        balanced_df['id'] = range(1, len(balanced_df) + 1)
        
        # Initialize peptide sequences
        balanced_df['peptide_seq'] = 'SYNTHETIC'
        balanced_df['peptide_len'] = 14
        
        # Copy original peptide sequences for the first samples
        original_count = len(df)
        for i in range(min(original_count, len(balanced_df))):
            balanced_df.loc[i, 'peptide_seq'] = df.loc[i, 'peptide_seq']
            balanced_df.loc[i, 'peptide_len'] = df.loc[i, 'peptide_len']
            balanced_df.loc[i, 'id'] = df.loc[i, 'id']
        
        # Generate synthetic peptide sequences for the additional samples
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        synthetic_count = 0
        avg_length = int(df['peptide_len'].mean())
        
        for i in range(original_count, len(balanced_df)):
            # Generate peptide length (with some variation)
            seq_length = max(5, min(20, avg_length + random.randint(-3, 3)))
            
            # Generate synthetic peptide sequence
            synthetic_seq = ''.join(random.choices(amino_acids, k=seq_length))
            
            balanced_df.loc[i, 'peptide_seq'] = synthetic_seq
            balanced_df.loc[i, 'peptide_len'] = seq_length
            synthetic_count += 1
        
        print(f"Generated {synthetic_count} synthetic peptide sequences")
        
        # Reorder columns to match original dataset
        balanced_df = balanced_df[df.columns.tolist()]
        
        # Save balanced dataset
        output_file = 'peptide_baza_balanced.csv'
        balanced_df.to_csv(output_file, sep=';', index=False)
        
        print(f"Balanced dataset saved to: {output_file}")
        print(f"Total samples in balanced dataset: {len(balanced_df)}")
        
        # Verify the saved file
        verification_df = pd.read_csv(output_file, sep=';')
        print("\nVerification of saved file:")
        print(f"  Total samples: {len(verification_df)}")
        print(f"  synthesis_flag=True: {sum(verification_df['synthesis_flag'] == True)}")
        print(f"  synthesis_flag=False: {sum(verification_df['synthesis_flag'] == False)}")
        
        return True
        
    except Exception as e:
        print(f"Error during SMOTE balancing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("SMOTE PEPTIDE DATASET BALANCING")
    print("="*60)
    
    success = create_balanced_peptide_dataset()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Balanced dataset created!")
        print("="*60)
        print("\nFiles created:")
        print("- peptide_baza_balanced.csv: Balanced dataset with synthetic samples")
        print("\nThe balanced dataset now contains equal numbers of samples for both classes:")
        print("- synthesis_flag=True (targetcol=1): Successfully synthesizable peptides")
        print("- synthesis_flag=False (targetcol=0): Non-synthesizable peptides (original + synthetic)")
    else:
        print("\n" + "="*60)
        print("FAILED: Could not create balanced dataset")
        print("="*60)
        sys.exit(1)
