import os
import pandas as pd
import re
import sys
import json
import random
import time
from datetime import timedelta

# Import the new universal recipe directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'recipes'))
import universal_recipe

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def setup_api_keys():
    """Ensure .env exists (omitted full logic for brevity, keeping your exact previous logic is fine here)"""
    # Keep your existing setup_api_key logic here...
    env_file = '.env'
    if not os.path.exists(env_file):
        print("No .env file found. Creating empty template.")
        with open(env_file, 'w') as f:
            f.write("NVIDIA_BUILD_API_KEY=\nOPENAI_API_KEY=\nCLAUDE_API_KEY=\nGEMINI_API_KEY=\nMISTRAL_API_KEY=\nPERPLEXITY_API_KEY=\n")
    return None

def load_models_from_csv(csv_path="recipes/models.csv"):
    """Reads models.csv and returns a list of dictionaries for models marked 'yes'"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} is missing. Please create it.")
    
    df = pd.read_csv(csv_path)
    active_models = df[df['tested'].str.lower() == 'yes'].to_dict('records')
    return active_models

def extract_language_pair_from_filename(filename):
    pattern = r'^([a-zA-Z]+)-([a-zA-Z]+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_processing_state(state_file="processing_state.json"):
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_processing_state(state, state_file="processing_state.json"):
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {str(e)}")

def get_common_sentence_ids(input_dir, sample_size=200, random_seed=42):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError("No CSV files found in input directory")
    
    first_file = os.path.join(input_dir, csv_files[0])
    first_df = pd.read_csv(first_file)
    all_sentence_ids = list(range(len(first_df)))
    random.seed(random_seed)
    selected_ids = random.sample(all_sentence_ids, min(sample_size, len(all_sentence_ids)))
    selected_ids.sort()
    return selected_ids

def load_sampled_data(input_dir, sentence_ids):
    samples = {}
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    for file in csv_files:
        input_path = os.path.join(input_dir, file)
        df = pd.read_csv(input_path)
        if len(df) > max(sentence_ids):
            samples[file] = df.iloc[sentence_ids].reset_index(drop=True)
        else:
            available_ids = [sid for sid in sentence_ids if sid < len(df)]
            samples[file] = df.iloc[available_ids].reset_index(drop=True)
    return samples

def run_translation_only(input_dir, output_dir, models, state):
    print("Running translation only...")
    print(f"Initial state: {len(state)} entries")
    
    sentence_ids = get_common_sentence_ids(input_dir, sample_size=200, random_seed=42)
    samples = load_sampled_data(input_dir, sentence_ids)
    
    total_tasks = 0
    completed_tasks = 0
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    # Calculate tasks
    for file in csv_files:
        source_lang, target_lang = extract_language_pair_from_filename(file)
        if not source_lang or not target_lang:
            continue
        for model in models:
            safe_model_name = model['model_id'].replace('/', '_')
            state_key = f"{source_lang}-{target_lang}/{file}/{safe_model_name}"
            if not state.get(state_key, {}).get('translation_completed', False):
                total_tasks += 1
    
    print(f"Total tasks to process: {total_tasks}")
    start_time = time.time()
    
    for file in csv_files:
        source_lang, target_lang = extract_language_pair_from_filename(file)
        if not source_lang or not target_lang:
            continue
            
        lang_pair_dir = os.path.join(output_dir, f"{source_lang}-{target_lang}")
        os.makedirs(lang_pair_dir, exist_ok=True)
        sampled_df = samples[file]
        
        for model in models:
            safe_model_name = model['model_id'].replace('/', '_')
            output_filename = f"{os.path.splitext(file)[0]}_{safe_model_name}.csv"
            output_path = os.path.join(lang_pair_dir, output_filename)
            
            state_key = f"{source_lang}-{target_lang}/{file}/{safe_model_name}"
            
            if state.get(state_key, {}).get('translation_completed', False):
                print(f"Skipping translation for {model['model_id']} on {file} - already completed")
                continue
            
            print(f"Processing {file} with model {model['model_id']} for {source_lang}-{target_lang}")
            
            try:
                # Use the new universal recipe
                result_df = universal_recipe.translation_only(
                    df=sampled_df, 
                    source_lang=source_lang, 
                    target_lang=target_lang,
                    model_id=model['model_id'],
                    provider=model['provider']
                )
                
                result_df.to_csv(output_path, index=False)
                
                if state_key not in state:
                    state[state_key] = {}
                state[state_key]['translation_completed'] = True
                state[state_key]['rows_processed'] = len(result_df)
                state[state_key]['timestamp'] = pd.Timestamp.now().isoformat()
                save_processing_state(state)
                
                completed_tasks += 1
                elapsed_time = time.time() - start_time
                if completed_tasks > 0:
                    time_per_task = elapsed_time / completed_tasks
                    eta_seconds = time_per_task * (total_tasks - completed_tasks)
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    progress = (completed_tasks / total_tasks) * 100
                    print(f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) - ETA: {eta_str}")
            except Exception as e:
                print(f"Error applying {model['model_id']} to {file}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nTranslation process completed in {timedelta(seconds=int(total_time))}! Final state: {len(state)} entries")

def main():
    # Make sure you populate your API keys correctly in the .env file
    setup_api_keys() 
    
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models directly from CSV
    try:
        models = load_models_from_csv("recipes/models.csv")
    except FileNotFoundError as e:
        print(e)
        return
        
    state = load_processing_state()
    
    run_translation_only(input_dir, output_dir, models, state)
    print("Translation completed! Now run reporting_combined.py to calculate similarity and generate reports.")

if __name__ == "__main__":
    main()
