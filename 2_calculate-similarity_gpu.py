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
OUTPUT_COMBINED_PATH = "output"
BATCH_SIZE = 256  
DEBUG_MODE = True
# ==============================================================================

def get_base_model_name(filename: str) -> str:
    """Strips the language prefix (e.g., 'ewe-eng_') from the CSV filename."""
    if '_' in filename:
        return filename.split('_', 1)[1]
    return filename

def validate_dataset_globally(csv_files: List[str]) -> List[str]:
    """
    Validates at the BASE MODEL level across all language folders:
    - A base model must exist in ALL language folders found.
    - A base model must not have any empty values in 'translated' or 'ref' columns.
    """
    print("\n" + "="*70)
    print("GLOBAL VALIDATION PHASE")
    print("="*70)
    
    # Group by BASE model
    model_files = defaultdict(list)
    all_langs = set()
    
    for file_path in csv_files:
        parent_dir = os.path.dirname(file_path)
        lang_pair = os.path.basename(parent_dir)
        filename = os.path.basename(file_path)
        base_model_name = get_base_model_name(filename)
        
        all_langs.add(lang_pair)
        model_files[base_model_name].append((lang_pair, file_path, filename))
        
    problematic_models = {}
    
    print(f"Checking {len(model_files)} base models across {len(all_langs)} language pairs...")
    
    for base_model_name, files in model_files.items():
        # 1. Check if the model exists in ALL language folders
        langs_for_this_model = set(lang for lang, _, _ in files)
        missing_langs = all_langs - langs_for_this_model
        
        if missing_langs:
            problematic_models[base_model_name] = f"Missing from language folder(s): {', '.join(missing_langs)}"
            continue
            
        # 2. Check for empty values in 'translated' or 'ref' columns
        for lang_pair, file_path, original_filename in files:
            try:
                df = pd.read_csv(file_path)
                if 'translated' not in df.columns or 'ref' not in df.columns:
                    problematic_models[base_model_name] = f"Missing 'translated' or 'ref' column in {original_filename}"
                    break
                    
                is_empty_trans = (df['translated'].isna() | 
                                  df['translated'].astype(str).str.strip().eq('') | 
                                  df['translated'].astype(str).str.lower().eq('nan'))
                
                is_empty_ref = (df['ref'].isna() | 
                                df['ref'].astype(str).str.strip().eq('') | 
                                df['ref'].astype(str).str.lower().eq('nan'))
                
                if is_empty_trans.any() or is_empty_ref.any():
                    problematic_models[base_model_name] = f"Empty 'translated' or 'ref' values found in {original_filename}"
                    break
                    
            except Exception as e:
                problematic_models[base_model_name] = f"Error reading {original_filename}: {e}"
                break

    if problematic_models:
        print(f"\n🚨 WARNING: Detected {len(problematic_models)} base model(s) failing strict validation:")
        for model, reason in problematic_models.items():
            print(f"  - {model}: {reason}")
            
        while True:
            response = input("\nDo you want to EXCLUDE these models entirely and proceed with the rest? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\nExcluding invalid models...")
                # Keep files only if their base model name is NOT in problematic_models
                valid_csv_files = [f for f in csv_files if get_base_model_name(os.path.basename(f)) not in problematic_models]
                return valid_csv_files
            elif response in ['n', 'no']:
                print("\nOperation aborted by user.")
                exit(0)
            else:
                print("Please enter 'y' to proceed or 'n' to abort.")
    else:
        print("✓ All models passed validation. Consistent across languages and fully populated.")
        return csv_files


if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True  
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print(f"Using device: {device} (Apple Silicon)")
else:
    device = "cpu"
    print(f"⚠ Warning: No GPU available, falling back to {device}")


def load_model_for_gpu():
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

    model.eval()
    model.half()  
    return model


def ensure_similarity_column(file_path: str, debug: bool = False) -> int:
    try:
        df = pd.read_csv(file_path)
        if 'similarity_score' not in df.columns:
            df['similarity_score'] = np.nan
            df.to_csv(file_path, index=False)
        return 0
    except Exception as e:
        print(f"  ✗ Error updating {file_path}: {e}")
        return 0

def collect_all_missing_pairs(csv_files: List[str], debug: bool = False) -> Tuple[List[Dict], Dict[str, int]]:
    print("\n" + "-"*70)
    print("PHASE 1: Scanning CSVs and collecting missing similarity pairs...")
    print("-"*70)
    
    all_pairs = []
    file_stats = defaultdict(int)
    
    for file_path in tqdm(csv_files, desc="Scanning CSVs"):
        ensure_similarity_column(file_path, debug)
        
        try:
            df = pd.read_csv(file_path)
            missing_mask = df['similarity_score'].isna() | (df['similarity_score'] == 0.0)
            missing_indices = df[missing_mask].index.tolist()
            
            if not missing_indices:
                file_stats['complete'] += 1
                continue
            
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
    return all_pairs, dict(file_stats)

def process_all_pairs_batch(pairs: List[Dict], similarity_model, batch_size: int = 256, debug: bool = False) -> List[Dict]:
    if not pairs:
        return []
    
    print("\n" + "-"*70)
    print("PHASE 2: Calculating similarity scores in centralized batches...")
    print("-"*70)
    
    all_translated = [p['translated'] for p in pairs]
    all_refs = [p['ref'] for p in pairs]
    
    all_similarities = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    
    process_start = time.time()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Calculating similarities", total=total_batches):
            batch_end = min(i + batch_size, len(pairs))
            batch_translated = all_translated[i:batch_end]
            batch_ref = all_refs[i:batch_end]
            
            embeddings_translated = similarity_model.encode(
                batch_translated, convert_to_tensor=True, device=device,
                normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
            )
            
            embeddings_ref = similarity_model.encode(
                batch_ref, convert_to_tensor=True, device=device,
                normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
            )
            
            batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
            batch_scores = batch_similarities.diag().cpu().numpy().astype(np.float32)
            
            all_similarities.extend(batch_scores)
            
            if i % (batch_size * 10) == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            
            del embeddings_translated, embeddings_ref, batch_similarities
    
    process_time = time.time() - process_start
    
    results = []
    for pair, score in zip(pairs, all_similarities):
        pair['similarity_score'] = float(score)
        results.append(pair)
    
    print(f"✓ Processed {len(results)} pairs in {process_time:.2f}s")
    return results

def update_csvs_with_results(results: List[Dict], debug: bool = False):
    print("\n" + "-"*70)
    print("PHASE 3: Updating CSV files with results...")
    print("-"*70)
    
    file_groups = defaultdict(list)
    for result in results:
        file_groups[result['file_path']].append(result)
    
    for file_path, file_results in tqdm(file_groups.items(), desc="Updating CSVs"):
        try:
            df = pd.read_csv(file_path)
            for result in file_results:
                idx = result['row_index']
                if idx in df.index:
                    df.loc[idx, 'similarity_score'] = result['similarity_score']
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"  ✗ Error updating {file_path}: {e}")

def find_csv_files(folder_path: str) -> List[str]:
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)  

def main():
    print("\n" + "="*70)
    print("SIMILARITY SCORE CALCULATOR - GPU OPTIMIZED (STRICT VALIDATION)")
    print("="*70)
    
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        print(f"\n✗ Error: Path does not exist: {OUTPUT_COMBINED_PATH}")
        return
    
    csv_files = find_csv_files(OUTPUT_COMBINED_PATH)
    if not csv_files:
        print("⚠ No CSV files found!")
        return
        
    valid_csv_files = validate_dataset_globally(csv_files)
    
    if not valid_csv_files:
        print("⚠ No valid CSV files remaining after validation. Exiting.")
        return
        
    all_pairs, stats = collect_all_missing_pairs(valid_csv_files, DEBUG_MODE)
    
    if not all_pairs:
        print("\n✓ All similarity scores are already calculated!")
        return
    
    similarity_model = load_model_for_gpu()
    results = process_all_pairs_batch(all_pairs, similarity_model, BATCH_SIZE, DEBUG_MODE)
    update_csvs_with_results(results, DEBUG_MODE)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total pairs processed: {len(results)}")
    print("✓ Processing complete!")
    print("="*70)
    
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
