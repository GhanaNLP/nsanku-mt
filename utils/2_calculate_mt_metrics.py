"""
calculate_mt_metrics.py
Calculate standard MT evaluation metrics: BLEU (corpus-level), chrF (sentence-level)
IMPORTANT: BLEU is calculated at corpus level for each file, not sentence-by-sentence
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import MT evaluation libraries
import sacrebleu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_COMBINED_PATH = "/home/owusus/Documents/GitHub/nsanku/output_combined"
DEBUG_MODE = True
# ==============================================================================

print("="*70)
print("MT METRICS CALCULATOR (BLEU corpus-level, chrF sentence-level)")
print("="*70)
print("\nIMPORTANT: BLEU is calculated at CORPUS LEVEL (proper evaluation)")
print("           chrF is calculated at sentence level")
print("="*70)

def calculate_chrf(hypothesis: str, reference: str) -> float:
    """Calculate chrF score for a single sentence pair"""
    try:
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
        return chrf.score
    except:
        return 0.0

def calculate_corpus_bleu(hypotheses: list, references: list) -> float:
    """
    Calculate corpus-level BLEU score (the correct way to evaluate BLEU)
    
    Args:
        hypotheses: list of translation strings
        references: list of reference strings
    
    Returns:
        BLEU score (0-100)
    """
    try:
        # Format references as list of lists (sacrebleu format)
        refs = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(hypotheses, refs)
        return bleu.score
    except:
        return 0.0

def process_file_metrics(file_path: str, debug: bool = False) -> bool:
    """
    Process a single CSV file:
    1. Calculate chrF at sentence level
    2. Calculate BLEU at corpus level for the entire file
    3. Assign the corpus BLEU score to all rows
    
    Returns True if successful, False otherwise
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'translated' not in df.columns or 'ref' not in df.columns:
            if debug:
                print(f"  Skipping {os.path.basename(file_path)}: missing required columns")
            return False
        
        # Add metric columns if missing
        if 'bleu_score' not in df.columns:
            df['bleu_score'] = np.nan
        if 'chrf_score' not in df.columns:
            df['chrf_score'] = np.nan
        if 'avg_score' not in df.columns:
            df['avg_score'] = np.nan
        
        # Check if we need to calculate anything
        needs_bleu = df['bleu_score'].isna().any()
        needs_chrf = df['chrf_score'].isna().any()
        
        if not needs_bleu and not needs_chrf:
            if debug:
                print(f"  ✓ {os.path.basename(file_path)}: Already complete")
            return True
        
        # Collect valid pairs
        valid_mask = (
            df['translated'].notna() & 
            df['ref'].notna() & 
            (df['translated'] != 'nan') & 
            (df['ref'] != 'nan') &
            (df['translated'].astype(str).str.strip() != '') &
            (df['ref'].astype(str).str.strip() != '')
        )
        
        if valid_mask.sum() == 0:
            if debug:
                print(f"  ⚠ {os.path.basename(file_path)}: No valid pairs")
            return False
        
        # CORPUS-LEVEL BLEU (the correct way!)
        if needs_bleu:
            hypotheses = df.loc[valid_mask, 'translated'].astype(str).str.strip().tolist()
            references = df.loc[valid_mask, 'ref'].astype(str).str.strip().tolist()
            
            corpus_bleu_score = calculate_corpus_bleu(hypotheses, references)
            
            # Assign the SAME corpus-level score to all valid rows
            df.loc[valid_mask, 'bleu_score'] = corpus_bleu_score
            # Set invalid rows to 0
            df.loc[~valid_mask, 'bleu_score'] = 0.0
            
            if debug:
                print(f"  BLEU (corpus): {corpus_bleu_score:.2f} for {os.path.basename(file_path)}")
        
        # SENTENCE-LEVEL chrF (this is fine to do sentence-by-sentence)
        if needs_chrf:
            missing_chrf_mask = valid_mask & df['chrf_score'].isna()
            
            if missing_chrf_mask.any():
                for idx in df[missing_chrf_mask].index:
                    translated = str(df.loc[idx, 'translated']).strip()
                    ref = str(df.loc[idx, 'ref']).strip()
                    df.loc[idx, 'chrf_score'] = calculate_chrf(translated, ref)
                
                # Set invalid rows to 0
                df.loc[~valid_mask & df['chrf_score'].isna(), 'chrf_score'] = 0.0
        
        # Calculate average
        df['avg_score'] = df[['bleu_score', 'chrf_score']].mean(axis=1)
        
        # Save
        df.to_csv(file_path, index=False)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def find_csv_files(folder_path: str) -> List[str]:
    """Find all CSV files recursively"""
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def main():
    print("\n" + "="*70)
    print("MT METRICS CALCULATOR - CORPUS-LEVEL BLEU")
    print("="*70)
    print(f"Output folder: {OUTPUT_COMBINED_PATH}")
    print(f"Metrics: BLEU (corpus-level), chrF (sentence-level)")
    print(f"Debug mode: {DEBUG_MODE}")
    print("="*70)
    print("\nIMPORTANT: BLEU is calculated at CORPUS level for each file.")
    print("This is the CORRECT way to evaluate BLEU scores!")
    print("All rows in a file get the same BLEU score (the corpus score).")
    print("="*70)
    
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        print(f"\n✗ Error: Path does not exist: {OUTPUT_COMBINED_PATH}")
        return
    
    # Find all CSV files
    print("\nScanning for CSV files...")
    csv_files = find_csv_files(OUTPUT_COMBINED_PATH)
    
    if not csv_files:
        print("⚠ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each file
    print("\n" + "-"*70)
    print("PROCESSING FILES (corpus-level BLEU, sentence-level chrF)")
    print("-"*70)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for file_path in tqdm(csv_files, desc="Processing files"):
        result = process_file_metrics(file_path, DEBUG_MODE)
        if result:
            success_count += 1
        elif result is False:
            error_count += 1
        else:
            skip_count += 1
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Files processed successfully: {success_count}")
    print(f"Files with errors: {error_count}")
    print(f"Total files: {len(csv_files)}")
    print(f"\nMetrics calculated:")
    print(f"  - BLEU: Corpus-level (same score for all rows in each file)")
    print(f"  - chrF: Sentence-level (individual score per row)")
    print(f"  - Average: Mean of BLEU and chrF")
    print("✓ Processing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
