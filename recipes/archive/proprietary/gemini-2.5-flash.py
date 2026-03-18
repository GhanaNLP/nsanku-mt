import pandas as pd
import time
import os
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Mock/Import utils - adjust this if your path differs
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from reporting import get_language_name
except ImportError:
    # Fallback if utility isn't found
    def get_language_name(code):
        return code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

def translate_text_with_gemini(text, source_lang, target_lang, max_retries=5):
    """
    Translate text using Gemini.
    Note: 'gemini-2.0-flash' is the standard high-speed model. 
    Thinking is NOT enabled by default on standard models.
    """
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    text = str(text).strip()
    if not text:
        return ""

    prompt = f"""Translate the following {source_lang_name} text into {target_lang_name}. 
Return ONLY the translation, no explanations or additional text.

Text to translate: {text}

Translation:"""

    # We initialize the model outside the loop for efficiency
    # Using 'gemini-1.5-flash' or 'gemini-2.0-flash-exp' depending on your access
    model_name = 'gemini-2.5-flash' 
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            
            # Use exponential backoff for retries
            wait_time = (2 ** attempt) + 1
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    top_p=0.9,
                    top_k=40,
                ),
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
                }
            )
            
            if not response or not response.candidates:
                logger.warning(f"No candidates in response on attempt {attempt+1}")
                time.sleep(wait_time)
                continue

            # Check if content was blocked
            if response.candidates[0].finish_reason == 3: # SAFETY
                logger.warning(f"Content blocked by safety filters for: {text[:30]}")
                # Optional: try a more relaxed retry or return empty
                return "[Blocked by Safety]"
            
            response_text = response.text
            
            if response_text:
                # Clean the response
                response_text = response_text.strip()
                # Remove common markdown artifacting if LLM ignores "ONLY translation" instruction
                response_text = re.sub(r'^[\[\]"\'‘’“”]+|[\[\]"\'‘’“”]+$', '', response_text)
                return response_text
                
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            if "429" in str(e): # Rate limit
                time.sleep(wait_time * 2)
            else:
                time.sleep(wait_time)
                
    return ""

def calculate_similarity(translated, reference):
    """Calculate cosine similarity between translated text and reference text"""
    try:
        if not translated or not reference or translated == "[Blocked by Safety]":
            return 0.0

        embeddings = similarity_model.encode([translated, reference])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def translation_only(df, source_lang, target_lang):
    """Only perform translation without similarity calculation"""
    result_df = df.copy()
    result_df['translated'] = ""

    # Rate limiting - adjust based on your tier (Free vs Pay-as-you-go)
    delay_between_requests = 1.0 

    total_texts = len(result_df)
    failed_translations = 0
    
    for i, row in result_df.iterrows():
        text = row['text']
        print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
        
        translation = translate_text_with_gemini(text, source_lang, target_lang)
        
        if translation:
            result_df.at[i, 'translated'] = translation
            print(f"  → {translation[:50]}...")
        else:
            failed_translations += 1
            print("  → [Translation failed]")
        
        if i < total_texts - 1:
            time.sleep(delay_between_requests)

    return result_df

def similarity_only(df, batch_size=32):
    """Batch processing for similarity"""
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
        similarities.extend(batch_sims.diag().tolist())
    
    result_df['similarity_score'] = similarities
    return result_df

def process_dataframe(df, source_lang, target_lang):
    df = translation_only(df, source_lang, target_lang)
    df = similarity_only(df)
    return df
