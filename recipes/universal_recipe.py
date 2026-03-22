import pandas as pd
import time
import os
import re
import requests
import concurrent.futures
import threading
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from typing import List


# Load environment variables
load_dotenv()


# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from reporting import get_language_name
except ImportError:
    def get_language_name(code):
        return code


# Initialize similarity model (kept for downstream compatibility)
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)


# --- Client Initializers ---
def get_openai_compatible_client(provider):
    if provider == "nvidia":
        return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_BUILD_API_KEY"))
    elif provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "mistral":
        return OpenAI(base_url="https://api.mistral.ai/v1", api_key=os.getenv("MISTRAL_API_KEY"))
    elif provider == "perplexity":
        return OpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PERPLEXITY_API_KEY"))
    elif provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    return None


def extract_bracketed_text(text):
    """Extract text from brackets if present (for LLMs instructed to return [text])"""
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def translate_llm(client, text, source_lang, target_lang, model_id, provider, max_retries=5):
    """Generic LLM translation handler"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_id,
                    max_tokens=2024,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
            elif provider == "gemini":
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt, generation_config={"temperature": 0.3})
                response_text = response.text
            else:
                # Standard OpenAI compatible format (NVIDIA, OpenAI, Mistral, Perplexity)
                api_kwargs = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }

                # Dynamically handle parameter requirements based on provider and model
                if provider == "openai":
                    # Newer OpenAI models use max_completion_tokens
                    api_kwargs["max_completion_tokens"] = 2024
                    # Reasoning models (o1, o3) typically reject temperature and top_p
                    if not any(r in model_id.lower() for r in ["o1", "o3"]):
                        api_kwargs["temperature"] = 0.3
                        api_kwargs["top_p"] = 0.95
                else:
                    # Legacy standard format for other providers
                    api_kwargs["max_tokens"] = 2024
                    api_kwargs["temperature"] = 0.3
                    api_kwargs["top_p"] = 0.95

                # Explicitly disable reasoning ONLY for known reasoning models on NVIDIA
                if provider == "nvidia":
                    reasoning_keywords = ["deepseek", "kimi", "nemotron", "reasoning", "think", "r1"]
                    if any(keyword in model_id.lower() for keyword in reasoning_keywords):
                        api_kwargs["extra_body"] = {
                            "chat_template_kwargs": {
                                "thinking": False
                            }
                        }

                completion = client.chat.completions.create(**api_kwargs)
                response_text = completion.choices[0].message.content

            return extract_bracketed_text(response_text)

        except Exception as e:
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Attempt {attempt+1} failed for text '{text[:20]}...': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                return ""


def translation_only(df, source_lang, target_lang, model_id, provider):
    """Perform translation dynamically based on provider"""
    print(f"Translation: {provider.upper()}")
    print(f"Model: {model_id}")

    result_df = df.copy()
    result_df['translated'] = ""
    total_texts = len(result_df)

    # BCP-47 mapping for NLLB models
    NLLB_LANG_MAP = {
        'en': 'eng_Latn',
        'eng': 'eng_Latn',
        'fr': 'fra_Latn',
        'es': 'spa_Latn',
        'de': 'deu_Latn',
        'it': 'ita_Latn',
        'zh': 'zho_Hans',
        'ar': 'arb_Arab',
        'pt': 'por_Latn',
        'ru': 'rus_Cyrl',
        'ja': 'jpn_Jpan',
        'ko': 'kor_Hang',
        'nl': 'nld_Latn',
        'pl': 'pol_Latn',
        'tr': 'tur_Latn',
        'vi': 'vie_Latn',
        'ewe': 'ewe_Latn',
        'twi': 'twi_Latn',
        'aka': 'aka_Latn',
    }

    # googletrans language code mapping
    GOOGLETRANS_LANG_MAP = {
        'en':  'en',
        'eng': 'en',
        'fr':  'fr',
        'es':  'es',
        'de':  'de',
        'it':  'it',
        'pt':  'pt',
        'ru':  'ru',
        'zh':  'zh-cn',
        'ar':  'ar',
        'ja':  'ja',
        'ko':  'ko',
        'nl':  'nl',
        'pl':  'pl',
        'tr':  'tr',
        'vi':  'vi',
        'ewe': 'ee',
        'twi': 'ak',
        'aka': 'ak',
        'gaa': 'gaa',
    }

    # 0. Handle NLLB via the high-throughput API
    if provider in ["nllb", "nllb-api", "nllb-ct2"]:
        src_bcp47 = NLLB_LANG_MAP.get(source_lang, f"{source_lang}_Latn")
        tgt_bcp47 = NLLB_LANG_MAP.get(target_lang, f"{target_lang}_Latn")

        api_base_url = os.getenv("NLLB_API_URL", "https://winstxnhdw-nllb-api.hf.space")
        endpoint = f"{api_base_url.rstrip('/')}/api/v4/translator"
        
        print(f"Routing NLLB translations through API: {api_base_url}")

        for i, row in result_df.iterrows():
            text = row['text']
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Translating {i+1}/{total_texts}: {text[:50]}...")
            
            try:
                params = {
                    "text": text,
                    "source": src_bcp47,
                    "target": tgt_bcp47
                }
                
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()
                
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        translation = data.get('translation', data.get('text', response.text))
                    else:
                        translation = str(data)
                except ValueError:
                    translation = response.text.strip()
                    
                if translation.startswith('"') and translation.endswith('"'):
                    translation = translation[1:-1]
                    
                result_df.at[i, 'translated'] = translation
                print(f"  → {translation[:50]}...")
                
            except requests.exceptions.RequestException as e:
                print(f"  → [API Error]: {e}")
                result_df.at[i, 'translated'] = ""

        return result_df

    # 1. Handle Opus-MT
    elif provider == "opus-mt":
        from transformers import pipeline
        import torch

        NLLB_BATCH_SIZE = 16 if torch.cuda.is_available() else 4

        try:
            translator = pipeline('translation', model=model_id)
        except Exception as e:
            print(f"Could not load Opus-MT model {model_id}: {e}")
            return result_df

        texts = result_df['text'].tolist()
        translations = [''] * total_texts

        for batch_start in range(0, total_texts, NLLB_BATCH_SIZE):
            batch_end = min(batch_start + NLLB_BATCH_SIZE, total_texts)
            batch = texts[batch_start:batch_end]
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Translating rows {batch_start+1}–{batch_end}/{total_texts}...")
            try:
                results = translator(batch, batch_size=len(batch))
                for j, res in enumerate(results):
                    if isinstance(res, list):
                        res = res[0]
                    translation = res.get('translation_text', '')
                    translations[batch_start + j] = translation
                    print(f"  [{batch_start+j+1}] → {translation[:50]}...")
            except Exception as e:
                print(f"  → [Batch failed]: {e}")
                for j, text in enumerate(batch):
                    try:
                        out = translator(text)[0]
                        if isinstance(out, list):
                            out = out[0]
                        translations[batch_start + j] = out.get('translation_text', '')
                    except Exception as e2:
                        print(f"  → [Row {batch_start+j+1} failed]: {e2}")

        result_df['translated'] = translations
        return result_df

    # 2. Handle Google Translate
    elif provider == "googletrans":
        from googletrans import Translator
        import asyncio
        import nest_asyncio

        nest_asyncio.apply()
        translator_client = Translator()

        gt_src = GOOGLETRANS_LANG_MAP.get(source_lang, source_lang)
        gt_tgt = GOOGLETRANS_LANG_MAP.get(target_lang, target_lang)
        print(f"Google Translate: {source_lang} → {gt_src} | {target_lang} → {gt_tgt}")

        async def fetch_translation(t, s, d):
            return await translator_client.translate(t, src=s, dest=d)

        for i, row in result_df.iterrows():
            text = row['text']
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Translating {i+1}/{total_texts}: {text[:50]}...")
            try:
                res = asyncio.run(fetch_translation(text, gt_src, gt_tgt))

                if isinstance(res, list):
                    result_df.at[i, 'translated'] = res[0].text
                else:
                    result_df.at[i, 'translated'] = res.text

                print(f"  → {result_df.at[i, 'translated'][:50]}...")
            except Exception as e:
                print(f"  → [Failed]: {e}")
            time.sleep(0.1)

        return result_df

    # 3. Handle APIs (LLMs) with Synchronized Batching
    else:
        client = get_openai_compatible_client(provider) if provider != "gemini" else None
        
        batch_size = 10           # Send 5 requests simultaneously
        pause_between_batches = 15 # Wait 5 seconds AFTER all 5 have finished
        stagger_delay = 3      # Brief wait between individual request submissions inside the batch
        
        df_lock = threading.Lock()
        
        print(f"\n--- Starting synchronized batch processing (Batches of {batch_size}, waiting for ALL to return, then {pause_between_batches}s pause) ---")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            
            # --- BACKGROUND COLLECTOR ---
            def collect_result(future, idx):
                try:
                    translation = future.result()
                    with df_lock:
                        result_df.at[idx, 'translated'] = translation
                    
                    ts = datetime.now().strftime('%H:%M:%S')
                    if translation:
                        print(f"[{ts}]  → [{idx+1}] Success: {translation[:50]}...")
                    else:
                        print(f"[{ts}]  → [{idx+1}] [Translation failed]")
                except Exception as e:
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts}]  → [{idx+1}] [Error]: {e}")
                    with df_lock:
                        result_df.at[idx, 'translated'] = ""

            # --- MAIN SUBMISSION LOOP ---
            # Loop through the dataframe in chunks of `batch_size`
            for i in range(0, total_texts, batch_size):
                batch_df = result_df.iloc[i:i+batch_size]
                
                print(f"\n--- Starting Batch {i//batch_size + 1} (Rows {i+1} to {min(i+batch_size, total_texts)}) ---")
                
                # Keep track of the active tasks for this specific batch
                current_batch_futures = []
                
                for idx, row in batch_df.iterrows():
                    text = row['text']
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts}] Submitting {idx+1}/{total_texts}: {text[:30]}...")
                    
                    # Submit the task to the pool
                    future = executor.submit(
                        translate_llm, client, text, source_lang, target_lang, model_id, provider
                    )
                    future.add_done_callback(lambda f, index=idx: collect_result(f, index))
                    
                    # Add to our tracker list
                    current_batch_futures.append(future)
                    
                    # Brief stagger so we don't spam 5 requests at the exact same millisecond
                    time.sleep(stagger_delay)
                
                # ---------------------------------------------------------
                # BLOCKING WAIT: Don't move forward until ALL requests 
                # in current_batch_futures have returned a response
                # ---------------------------------------------------------
                concurrent.futures.wait(current_batch_futures)
                
                # Once wait() is done, the entire batch is finished. 
                # Now we sleep for 5 seconds (unless it's the very last batch).
                if i + batch_size < total_texts:
                    ts_pause = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts_pause}] --- Batch complete. All responses received. Pausing for {pause_between_batches} seconds ---")
                    time.sleep(pause_between_batches)
                    
        print("\nAll batches submitted and all results collected!")
        return result_df


# Retained downstream compatibility functions
def similarity_only(df, batch_size=32):
    print("Calculating similarity scores...")
    result_df = df.copy()
    if 'translated' not in result_df.columns or 'ref' not in result_df.columns:
        return result_df

    translated_texts = result_df['translated'].fillna('').tolist()
    ref_texts = result_df['ref'].fillna('').tolist()
    similarities = []

    for i in range(0, len(translated_texts), batch_size):
        batch_trans = translated_texts[i:i+batch_size]
        batch_ref = ref_texts[i:i+batch_size]
        emb_trans = similarity_model.encode(batch_trans, convert_to_tensor=True)
        emb_ref = similarity_model.encode(batch_ref, convert_to_tensor=True)
        batch_sims = util.pytorch_cos_sim(emb_trans, emb_ref)
        similarities.extend(batch_sims.diag().cpu().numpy())

    result_df['similarity_score'] = similarities
    return result_df


def process_dataframe(df, source_lang, target_lang, model_id, provider):
    df = translation_only(df, source_lang, target_lang, model_id, provider)
    df = similarity_only(df)
    return df
