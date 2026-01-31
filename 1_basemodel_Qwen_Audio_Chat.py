# evaluate_qwen_mmau.py
"""
Evaluate Qwen Audio Models on MMAU Benchmark
Using the correct Qwen-Audio API with tokenizer.from_list_format()
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import os
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path("/media/volume/h100_instance2/LALMs_Reasoning_Steering/MMAU")
AUDIO_BASE_DIR = BASE_DIR / "data/test-mini-audios/"
TEST_JSON = BASE_DIR / "mmau-test-mini.json"
EVAL_SCRIPT = BASE_DIR / "evaluation.py"
CACHE_DIR = "/media/volume/h100_instance2/cache"

# Set cache environment variables
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Helper Functions
# =============================================================================

def load_audio_path(item, audio_base_dir):
    """
    Find the correct audio file path from the item.
    
    Args:
        item: JSON item from MMAU dataset
        audio_base_dir: Base directory containing audio files
    
    Returns:
        Full path to audio file
    """
    audio_id = item.get('audio_id', '')
    
    # Clean the path
    audio_id = audio_id.replace('\\', '/').lstrip('./')
    
    # Try direct path
    audio_path = audio_base_dir / audio_id
    if audio_path.exists():
        return str(audio_path)
    
    # Try just the filename
    filename = Path(audio_id).name
    audio_path = audio_base_dir / filename
    if audio_path.exists():
        return str(audio_path)
    
    # Try with test-mini-audios subfolder
    audio_path = audio_base_dir / "test-mini-audios" / filename
    if audio_path.exists():
        return str(audio_path)
    
    raise FileNotFoundError(f"Could not find audio file: {audio_id}")

# =============================================================================
# Model Evaluation Function
# =============================================================================

def evaluate_qwen_model(model_name, output_file):
    """
    Evaluate a Qwen audio model on MMAU test-mini.
    
    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen-Audio-Chat")
        output_file: Path to save predictions JSON
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load Model and Tokenizer
    # -------------------------------------------------------------------------
    print("Step 1: Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, cache_dir=CACHE_DIR).eval()
    print(f"Model loaded on {model.device}")
    
    # -------------------------------------------------------------------------
    # Step 2: Load Test Data
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading test data...")
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")
    
    # -------------------------------------------------------------------------
    # Step 3: Run Inference
    # -------------------------------------------------------------------------
    print("\nStep 3: Running inference...")
    results = []
    correct = 0
    errors = 0
    
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        # Get audio path
        audio_path = load_audio_path(item, AUDIO_BASE_DIR)
        
        # Get question
        question = item.get('question', '')
        
        # Format the question as a multiple-choice prompt
        choices = item['choices']
        formatted_question = f"""{question}  Select one option from the provided choices.\n{choices}."""

        # print("Original question:", question)
        # print("Formatted question:", formatted_question)
        # print("Audio Path:", audio_path)
        # 1st dialogue turn
        # Format query using Qwen's list format
        query = tokenizer.from_list_format([
                {'audio': audio_path},
                {'text': formatted_question},
            ])
        
        response, history = model.chat(tokenizer, query=query, history=None)
        # Add model_prediction to the item
        item['model_output'] = response
        results.append(item)
    
    # -------------------------------------------------------------------------
    # Step 4: Save Results
    # -------------------------------------------------------------------------
    print(f"\nStep 4: Saving results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to: {output_file}")
    
    # -------------------------------------------------------------------------
    # Step 5: Print Summary
    # -------------------------------------------------------------------------
    total_answered = len(data) - errors
    print(f"\n{'='*70}")
    print(f"Generation Summary")
    print(f"{'='*70}")
    print(f"Model:           {model_name}")
    print(f"Total Samples:   {len(data)}")
    print(f"Answered:        {total_answered}")
    print(f"Errors:          {errors}")

# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    """Evaluate Qwen audio models"""
    
    print(f"\n{'='*70}")
    print(f"MMAU Evaluation - Qwen Audio Models")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"{'='*70}\n")
    
    models_to_evaluate = [
        # ("Qwen/Qwen-Audio-Chat", "qwen_audio_chat_results.json"),
        ("Qwen/Qwen2-Audio-7B", "qwen2_audio_results.json"),
        ("Qwen/Qwen2-Audio-7B-Instruct", "qwen2_audio_instruct_results.json")
    ]

    all_results = []
    
    for model_name, output_file in models_to_evaluate:
        try:
            evaluate_qwen_model(model_name, output_file)
        except Exception as e:
            print(f"\nFAILED to evaluate {model_name}: {e}\n")

# python 2_evaluation.py --input '/media/volume/h100_instance2/LALMs_Reasoning_Steering/qwen_audio_chat_results.json'
if __name__ == "__main__":
    main()