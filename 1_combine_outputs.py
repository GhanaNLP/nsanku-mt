import os
import pandas as pd
from collections import defaultdict
import zipfile
import tempfile

def combine_matching_csvs_from_zips(root_folder, output_path=None):
    """
    Combine model CSV files from language folders inside zip files.
    Validates at the MODEL (CSV filename) level:
    - A model must exist in ALL language folders found across the dataset.
    - A model must not have any empty values in the 'translated' column in ANY zip file.
    """
    
    # Set output directory
    if output_path is None:
        output_dir = os.path.join(root_folder, "output_combined")
    else:
        output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}")
    
    # Dictionary to group files by MODEL (CSV filename)
    # Format: model_files["model_A.csv"] = [(lang_pair, zip_file, full_path), ...]
    model_files = defaultdict(list)
    all_langs = set()
    
    # Find all zip files in the root folder
    zip_files = [f for f in os.listdir(root_folder) if f.endswith('.zip')]
    
    if not zip_files:
        print("No zip files found in the root folder!")
        return output_dir
    
    print(f"Found {len(zip_files)} zip files to process: {zip_files}")
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each zip file and collect all CSV files
        for zip_file in zip_files:
            zip_path = os.path.join(root_folder, zip_file)
            print(f"\nProcessing zip file: {zip_file}")
            
            try:
                extract_path = os.path.join(temp_dir, os.path.splitext(zip_file)[0])
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"  ✓ Extracted to temporary location")
                
                # Walk through the extracted content to find CSV files
                for foldername, subfolders, filenames in os.walk(extract_path):
                    if foldername == extract_path:
                        continue
                        
                    # The immediate subfolder is the language pair
                    lang_pair = os.path.basename(foldername)
                    all_langs.add(lang_pair)
                    
                    for filename in filenames:
                        if filename.endswith('.csv'):
                            full_path = os.path.join(foldername, filename)
                            # Group by the CSV name (which is the model name)
                            model_files[filename].append((lang_pair, zip_file, full_path))
                            print(f"  - Found: {lang_pair}/{filename} from {zip_file}")
                            
            except Exception as e:
                print(f"  ✗ Error processing {zip_file}: {e}")

        # ==========================================
        # VALIDATION PHASE: Model presence & Empty 'translated' column
        # ==========================================
        print(f"\n{'='*60}")
        print("VALIDATING MODELS (CSV FILES)")
        print(f"{'='*60}")
        
        problematic_models = {}
        
        for csv_name, files in model_files.items():
            # 1. Check if the model exists in ALL language folders
            langs_for_this_model = set(lang for lang, _, _ in files)
            missing_langs = all_langs - langs_for_this_model
            
            if missing_langs:
                problematic_models[csv_name] = f"Missing entirely from language folder(s): {', '.join(missing_langs)}"
                continue # Skip the translation check, we already know we need to flag this model
                
            # 2. Check for empty values in the 'translated' column across all its files in all zips
            for lang_pair, zip_source, file_path in files:
                try:
                    df = pd.read_csv(file_path)
                    if 'translated' in df.columns:
                        is_empty = df['translated'].isna() | df['translated'].astype(str).str.strip().eq('')
                        if is_empty.any():
                            problematic_models[csv_name] = f"Empty 'translated' values in {lang_pair} (Found in: {zip_source})"
                            break # No need to keep checking this model, flag it and move on
                    else:
                        problematic_models[csv_name] = f"Missing 'translated' column in {lang_pair} (Found in: {zip_source})"
                        break
                except Exception as e:
                    problematic_models[csv_name] = f"Error reading {lang_pair} from {zip_source}: {e}"
                    break

        # Prompt user if problematic models are found
        if problematic_models:
            print(f"\n🚨 WARNING: Detected {len(problematic_models)} model(s) failing validation:")
            for model, reason in problematic_models.items():
                print(f"  - {model}: {reason}")
                
            while True:
                response = input("\nDo you want to EXCLUDE these models across ALL languages and proceed? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    print("\nExcluding invalid models and proceeding...")
                    # Completely delete the flagged models from our merging dictionary
                    for model in problematic_models.keys():
                        if model in model_files:
                            del model_files[model]
                    break
                elif response in ['n', 'no']:
                    print("\nOperation aborted by user. No files were combined.")
                    return output_dir
                else:
                    print("Please enter 'y' to proceed or 'n' to abort.")
        else:
            print("  ✓ All models passed validation. Consistent across languages and fully translated.")

        # ==========================================
        # COMBINATION PHASE
        # ==========================================
        if not model_files:
            print("\nNo valid models left to combine after filtering. Exiting.")
            return output_dir

        print(f"\n{'='*60}")
        print("COMBINING VALID MODELS")
        print(f"{'='*60}")
        
        total_combined = 0
        
        for csv_name, files in model_files.items():
            print(f"\n🤖 Processing Model: {csv_name}")
            
            # Re-group the files for this model by language pair
            lang_groups = defaultdict(list)
            for lang_pair, zip_source, file_path in files:
                lang_groups[lang_pair].append((file_path, zip_source))
                
            for lang_pair, file_info in lang_groups.items():
                lang_pair_output_dir = os.path.join(output_dir, lang_pair)
                os.makedirs(lang_pair_output_dir, exist_ok=True)
                output_file_path = os.path.join(lang_pair_output_dir, csv_name)
                
                dataframes = []
                for file_path, zip_source in file_info:
                    try:
                        df = pd.read_csv(file_path)
                        df['source_zip_file'] = os.path.splitext(zip_source)[0]
                        dataframes.append(df)
                    except Exception as e:
                        print(f"  ✗ Error reading {file_path}: {e}")
                
                if dataframes:
                    combined_df = pd.concat(dataframes, ignore_index=True)
                    combined_df.to_csv(output_file_path, index=False)
                    total_rows = len(combined_df)
                    total_sources = len(dataframes)
                    print(f"  ✅ Combined {lang_pair}: {total_rows} rows from {total_sources} zips")
                    total_combined += 1
                    
    print(f"\n{'='*60}")
    print("COMBINATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total language/model combinations created: {total_combined}")
    print(f"Combined data available in: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir

# Example usage
if __name__ == "__main__":
    # Specify your paths here
    root_path = "output"  # Folder containing the zip files
    
    combine_matching_csvs_from_zips(root_path, "output_combined")
