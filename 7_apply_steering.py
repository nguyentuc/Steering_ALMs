def create_steering_hook(steering_vector, alpha=1.0):
    """
    Create hook function that applies steering
    
    Args:
        steering_vector: Tensor (hidden_dim,) - direction to steer
        alpha: Float - steering strength
    
    Returns:
        hook_fn: Function to register as forward hook
    """
    
    def hook_fn(module, input, output):
        """
        Hook function applied during forward pass
        
        Args:
            output: Hidden states from layer (batch_size, seq_len, hidden_dim)
        
        Returns:
            steered_output: output + alpha * steering_vector
        """
        
        # Move steering vector to same device as output
        steer_vec = steering_vector.to(output.device).to(output.dtype)
        
        # Expand steering vector to match output shape
        # From (hidden_dim,) to (1, 1, hidden_dim)
        # Then broadcast to (batch_size, seq_len, hidden_dim)
        steer_vec_expanded = steer_vec.unsqueeze(0).unsqueeze(0)
        
        # Apply steering
        steered_output = output + alpha * steer_vec_expanded
        
        return steered_output
    
    return hook_fn


def apply_steering(model, steering_vectors, alpha=1.0):
    """
    Apply steering to model by registering hooks
    
    Args:
        model: ALM to steer
        steering_vectors: Dict[layer_idx] = steering vector
        alpha: Steering strength
    
    Returns:
        model: Model with hooks registered
        handles: List of hook handles (for removal)
    """
    
    handles = []
    
    print(f"\n=== Applying Steering (alpha={alpha}) ===")
    
    for layer_idx, steering_vec in steering_vectors.items():
        # Create hook for this layer
        hook_fn = create_steering_hook(steering_vec, alpha)
        
        # Register hook
        layer_module = model.model.layers[layer_idx]
        handle = layer_module.register_forward_hook(hook_fn)
        handles.append(handle)
        
        print(f"✓ Layer {layer_idx}: Steering registered")
    
    print(f"✓ Total hooks registered: {len(handles)}")
    
    return model, handles


def remove_steering(handles):
    """Remove all steering hooks"""
    for handle in handles:
        handle.remove()
    print(f"✓ Removed {len(handles)} steering hooks")
    

## Apply steering vector
def evaluate_with_steering(
    model, 
    processor, 
    test_set, 
    steering_vectors,
    alpha=1.0,
    dataset_name="Test"
):
    """
    Evaluate model on test set with steering applied
    
    Returns:
        results: List of predictions
        accuracy: Overall accuracy
    """
    
    # Apply steering
    model, handles = apply_steering(model, steering_vectors, alpha)
    
    results = []
    correct = 0
    
    print(f"\n=== Evaluating on {dataset_name} (alpha={alpha}) ===")
    
    for sample in tqdm(test_set, desc="Evaluation"):
        # Load audio
        audio_path = sample['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        # Prepare prompt
        question = sample['question']
        choices = sample.get('choices', sample.get('options', []))
        
        choices_text = '\n'.join([f"{chr(65+i)}. {choice}" 
                                  for i, choice in enumerate(choices)])
        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        
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
                max_new_tokens=10,
                do_sample=False
            )
        
        # Decode
        prediction = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()
        
        pred_letter = prediction[0] if prediction else 'A'
        
        # Ground truth
        ground_truth = sample.get('answer', sample.get('ground_truth'))
        if ground_truth in choices:
            gt_letter = chr(65 + choices.index(ground_truth))
        else:
            gt_letter = ground_truth[0] if ground_truth else 'A'
        
        # Check correctness
        is_correct = (pred_letter == gt_letter)
        if is_correct:
            correct += 1
        
        results.append({
            'sample_id': sample.get('id', sample.get('sample_id')),
            'prediction': pred_letter,
            'ground_truth': gt_letter,
            'is_correct': is_correct,
            'difficulty': sample.get('difficulty', 'unknown'),
            'modality': sample.get('modality', sample.get('domain', 'unknown')),
            'category': sample.get('category', 'unknown')
        })
    
    # Remove steering
    remove_steering(handles)
    
    # Compute accuracy
    accuracy = correct / len(test_set) * 100
    
    print(f"\n{dataset_name} Results (alpha={alpha}):")
    print(f"  Correct: {correct}/{len(test_set)}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return results, accuracy