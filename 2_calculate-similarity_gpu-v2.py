import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import gc
import time
from collections import defaultdict
from typing import List, Dict, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_COMBINED_PATH = "output_combined"
BATCH_SIZE = 256  # Increased for GPU (much faster processing)
DEBUG_MODE = True
# ==============================================================================

def validate_dataset_globally(csv_files: List[str]) -> List[str]:
    """
    Validates at the MODEL (CSV filename) level across all language folders:
    - A model must exist in ALL language folders found.
    - A model must not have any empty values in 'translated' or 'ref' columns.
    """
    print("\n" + "="*70)
    print("GLOBAL VALIDATION PHASE")
    print("="*70)
    
    # Group by model (filename)
    # Format: model_files["model_A.csv"] = [(lang_pair, file_path), ...]
    model_files = defaultdict(list)
    all_langs = set()
    
    for file_path in csv_files:
        parent_dir = os.path.dirname(file_path)
        lang_pair = os.path.basename(parent_dir)
        model_name = os.path.basename(file_path)
        
        all_langs.add(lang_pair)
        model_files[model_name].append((lang_pair, file_path))
        
    problematic_models = {}
    
    print(f"Checking {len(model_files)} models across {len(all_langs)} language pairs...")
    
    for model_name, files in model_files.items():
        # 1. Check if the model exists in ALL language folders
        langs_for_this_model = set(lang for lang, _ in files)
        missing_langs = all_langs - langs_for_this_model
        
        if missing_langs:
            problematic_models[model_name] = f"Missing from language folder(s): {', '.join(missing_langs)}"
            continue
            
        # 2. Check for empty values in 'translated' or 'ref' columns
        for lang_pair, file_path in files:
            try:
                df = pd.read_csv(file_path)
                if 'translated' not in df.columns or 'ref' not in df.columns:
                    problematic_models[model_name] = f"Missing 'translated' or 'ref' column in {lang_pair}"
                    break
                    
                # Check translated
                is_empty_trans = (df['translated'].isna() | 
                                  df['translated'].astype(str).str.strip().eq('') | 
                                  df['translated'].astype(str).str.lower().eq('nan'))
                
                # Check ref (since similarity needs both)
                is_empty_ref = (df['ref'].isna() | 
                                df['ref'].astype(str).str.strip().eq('') | 
                                df['ref'].astype(str).str.lower().eq('nan'))
                
                if is_empty_trans.any() or is_empty_ref.any():
                    problematic_models[model_name] = f"Empty 'translated' or 'ref' values found in {lang_pair}"
                    break
                    
            except Exception as e:
                problematic_models[model_name] = f"Error reading {lang_pair}: {e}"
                break

    # Prompt user if problematic models are found
    if problematic_models:
        print(f"\n🚨 WARNING: Detected {len(problematic_models)} model(s) failing strict validation:")
        for model, reason in problematic_models.items():
            print(f"  - {model}: {reason}")
            
        while True:
            response = input("\nDo you want to EXCLUDE these models entirely and proceed with the rest? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\nExcluding invalid models...")
                # Filter the original csv_files list to only include valid models
                valid_csv_files = [f for f in csv_files if os.path.basename(f) not in problematic_models]
                return valid_csv_files
            elif response in ['n', 'no']:
                print("\nOperation aborted by user.")
                exit(0)
            else:
                print("Please enter 'y' to proceed or 'n' to abort.")
    else:
        print("✓ All models passed validation. Consistent across languages and fully populated.")
        return csv_files


# Auto-detect best available device
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"Using device: {device} (Apple Silicon)")
else:
    device = "cpu"
    print(f"⚠ Warning: No GPU available, falling back to {device}")


def load_model_for_gpu():
    """Wrapped model loading in a function so it only triggers if validation passes."""
    similarity_model_path = "sentence-transformers/all-mpnet-base-v2"
    print("\nLoading MPNet model for GPU...")

    model_load_start = time.time()
    try:
        model = SentenceTransformer(
            similarity_model_path,
            device=device,
            cache_folder="./model_cache",
            trust_remote_code=False
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Load failed: {e}")
        print("Attempting offline mode...")
        try:
            model = SentenceTransformer(
                similarity_model_path,
                device=device,
                cache_folder="./model_cache",
                local_files_only=True
            )
            print("✓ Model loaded from local cache")
        except Exception as e2:
            print(f"⚠ Offline load failed: {e2}")
            exit(1)

    model_load_time = time.time() - model_load_start
    print(f"✓ Model loaded in {model_load_time:.2f} seconds")

    # Configure model for GPU inference
    model.eval()
    model.half()  # Use FP16 for faster GPU inference
    print("✓ Model configured for FP16 inference on GPU")
    return model


def ensure_similarity_column(file_path: str, debug: bool = False) -> int:
    """
    Ensures the similarity_score column exists. 
    (Row dropping logic removed because global validation guarantees no empty rows).
    """
    try:
        df = pd.read_csv(file_path)
        if 'similarity_score' not in df.columns:
            df['similarity_score'] = np.nan
            df.to_csv(file_path, index=False)
            if debug:
                print(f"  Added similarity_score column to {os.path.basename(file_path)}")
        return 0
    except Exception as e:
        print(f"  ✗ Error updating {file_path}: {e}")
        return 0

def collect_all_missing_pairs(csv_files: List[str], debug: bool = False) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Phase 1: Collect all missing similarity pairs across all CSVs
    Returns list of pairs with metadata and file statistics
    """
    print("\n" + "-"*70)
    print("PHASE 1: Scanning CSVs and collecting missing similarity pairs...")
    print("-"*70)
    
    all_pairs = []
    file_stats = defaultdict(int)
    
    for file_path in tqdm(csv_files, desc="Scanning CSVs"):
        # Ensure column exists
        ensure_similarity_column(file_path, debug)
        
        try:
            df = pd.read_csv(file_path)
            
            # Find missing indices
            missing_mask = df['similarity_score'].isna() | (df['similarity_score'] == 0.0)
            missing_indices = df[missing_mask].index.tolist()
            
            if not missing_indices:
                file_stats['complete'] += 1
                continue
            
            # Collect pairs (now all pairs are guaranteed to be valid)
            for idx in missing_indices:
                translated = str(df.loc[idx, 'translated']).strip()
                ref = str(df.loc[idx, 'ref']).strip()
                
                all_pairs.append({
                    'file_path': file_path,
                    'row_index': idx,
                    'translated': translated,
                    'ref': ref
                })
            
            file_stats['pending'] += len(missing_indices)
            
        except Exception as e:
            print(f"  ✗ Error reading {file_path}: {e}")
    
    total_pairs = len(all_pairs)
    print(f"\n✓ Found {total_pairs} pairs needing calculation across {len(csv_files)} files")
    print(f"  Complete files: {file_stats['complete']}")
    print(f"  Pending calculations: {file_stats['pending']}")
    
    return all_pairs, dict(file_stats)

def process_all_pairs_batch(pairs: List[Dict], similarity_model, batch_size: int = 256, debug: bool = False) -> List[Dict]:
    """
    Phase 2: Process all pairs in centralized batches for maximum efficiency
    """
    if not pairs:
        return []
    
    print("\n" + "-"*70)
    print("PHASE 2: Calculating similarity scores in centralized batches...")
    print("-"*70)
    
    print(f"Valid pairs to process: {len(pairs)}")
    
    # Extract texts for batch processing
    all_translated = [p['translated'] for p in pairs]
    all_refs = [p['ref'] for p in pairs]
    
    # Process in batches
    all_similarities = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    
    process_start = time.time()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), 
                      desc="Calculating similarities", 
                      total=total_batches):
            batch_end = min(i + batch_size, len(pairs))
            batch_translated = all_translated[i:batch_end]
            batch_ref = all_refs[i:batch_end]
            
            # Encode both batches
            embeddings_translated = similarity_model.encode(
                batch_translated,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            
            embeddings_ref = similarity_model.encode(
                batch_ref,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            
            # Calculate similarities on GPU
            batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
            batch_scores = batch_similarities.diag().cpu().numpy().astype(np.float32)
            
            all_similarities.extend(batch_scores)
            
            # Periodic GPU memory cleanup
            if i % (batch_size * 10) == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            
            del embeddings_translated, embeddings_ref, batch_similarities
    
    process_time = time.time() - process_start
    
    # Combine results
    results = []
    for pair, score in zip(pairs, all_similarities):
        pair['similarity_score'] = float(score)
        results.append(pair)
    
    print(f"✓ Processed {len(results)} pairs in {process_time:.2f}s")
    print(f"  Average: {process_time/len(pairs):.4f}s per pair")
    
    return results

def update_csvs_with_results(results: List[Dict], debug: bool = False):
    """
    Phase 3: Update all CSVs with calculated scores
    """
    print("\n" + "-"*70)
    print("PHASE 3: Updating CSV files with results...")
    print("-"*70)
    
    # Group results by file
    file_groups = defaultdict(list)
    for result in results:
        file_groups[result['file_path']].append(result)
    
    print(f"Updating {len(file_groups)} files...")
    
    for file_path, file_results in tqdm(file_groups.items(), desc="Updating CSVs"):
        try:
            df = pd.read_csv(file_path)
            
            # Update rows
            update_count = 0
            for result in file_results:
                idx = result['row_index']
                if idx in df.index:
                    df.loc[idx, 'similarity_score'] = result['similarity_score']
                    update_count += 1
            
            # Save once per file
            df.to_csv(file_path, index=False)
            
            if debug:
                print(f"  Updated {update_count} rows in {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"  ✗ Error updating {file_path}: {e}")

def find_csv_files(folder_path: str) -> List[str]:
    """Find all CSV files recursively"""
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)  # Sort for consistent processing order

def main():
    print("\n" + "="*70)
    print("SIMILARITY SCORE CALCULATOR - GPU OPTIMIZED (STRICT VALIDATION)")
    print("="*70)
    print(f"Output folder: {OUTPUT_COMBINED_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Debug mode: {DEBUG_MODE}")
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
    
    # =================================================================
    # NEW: Run Global Validation BEFORE loading model or processing
    # =================================================================
    valid_csv_files = validate_dataset_globally(csv_files)
    
    if not valid_csv_files:
        print("⚠ No valid CSV files remaining after validation. Exiting.")
        return
        
    # Phase 1: Scan CSVs and collect all missing pairs
    all_pairs, stats = collect_all_missing_pairs(valid_csv_files, DEBUG_MODE)
    
    if not all_pairs:
        print("\n✓ All similarity scores are already calculated!")
        return
    
    # Only load the heavy GPU model if we actually have valid pairs to process
    similarity_model = load_model_for_gpu()
    
    # Phase 2: Process all pairs
    results = process_all_pairs_batch(all_pairs, similarity_model, BATCH_SIZE, DEBUG_MODE)
    
    # Phase 3: Update all CSVs
    update_csvs_with_results(results, DEBUG_MODE)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total pairs processed: {len(results)}")
    print(f"Files updated: {len(set(r['file_path'] for r in results))}")
    print("✓ Processing complete!")
    print("="*70)
    
    # GPU cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
