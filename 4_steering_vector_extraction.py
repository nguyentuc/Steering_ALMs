import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torchaudio
from tqdm import tqdm

def setup_model():
    """Initialize Qwen2-Audio-7B-Instruct"""
    
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    
    print("Loading model...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Model loaded")
    return model, processor


def run_inference_on_extraction_set(model, processor, extraction_set, dataset_name):
    """
    Run inference on extraction set and record results
    
    Returns:
        results: List of dicts with predictions and correctness
    """
    
    results = []
    
    print(f"\n=== Running inference on {dataset_name} extraction set ===")
    
    for sample in tqdm(extraction_set, desc="Inference"):
        # Load audio
        audio_path = sample['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        # Prepare inputs
        question = sample['question']
        choices = sample.get('choices', sample.get('options', []))
        
        # Format prompt (multiple choice)
        choices_text = '\n'.join([f"{chr(65+i)}. {choice}" 
                                  for i, choice in enumerate(choices)])
        
        prompt = f"""Question: {question}

Choices:
{choices_text}

Answer with just the letter (A, B, C, or D):"""
        
        # Generate prediction
        inputs = processor(
            text=[prompt],
            audios=[audio],
            return_tensors="pt",
            sampling_rate=sr
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,  # Just need single letter
                do_sample=False
            )
        
        # Decode prediction
        prediction = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()
        
        # Extract letter answer (A, B, C, D)
        pred_letter = prediction[0] if prediction else 'A'
        
        # Get ground truth
        ground_truth = sample.get('answer', sample.get('ground_truth'))
        
        # Convert ground truth to letter if needed
        if ground_truth in choices:
            gt_letter = chr(65 + choices.index(ground_truth))
        else:
            gt_letter = ground_truth[0] if ground_truth else 'A'
        
        # Check correctness
        is_correct = (pred_letter == gt_letter)
        
        # Store result
        results.append({
            'sample_id': sample.get('id', sample.get('sample_id')),
            'audio_path': audio_path,
            'question': question,
            'choices': choices,
            'prediction': pred_letter,
            'ground_truth': gt_letter,
            'is_correct': is_correct,
            'difficulty': sample.get('difficulty', 'unknown'),
            'modality': sample.get('modality', sample.get('domain', 'unknown')),
            'category': sample.get('category', 'unknown')
        })
    
    # Print statistics
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total * 100
    
    print(f"\n{dataset_name} Extraction Set Results:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return results


# Setup model
model, processor = setup_model()

# Run inference on MMAU extraction set (4,000 samples)
mmau_extraction_results = run_inference_on_extraction_set(
    model, processor, mmau_extraction, "MMAU"
)

# Save results
with open('extraction_inference_results.pkl', 'wb') as f:
    pickle.dump({
        'mmau': mmau_extraction_results
    }, f)

print("\n✓ Inference results saved")