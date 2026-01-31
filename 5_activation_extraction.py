def collect_residual_activations(model, processor, samples, target_layers=[8, 16, 24, 31]):
    """
    Collect activations from residual stream at target layers
    
    Key: Hook into the OUTPUT of each layer (after residual connection)
    
    Returns:
        activations: Dict[sample_id][layer_idx] = tensor (seq_len, hidden_dim)
    """
    
    activations = {}
    current_sample_id = None
    
    # Register hooks for target layers
    handles = []
    
    for layer_idx in target_layers:
        def hook_fn(module, input, output, layer=layer_idx):
            """
            Hook function to capture residual stream
            
            Args:
                output: Hidden states AFTER layer processing
                        Shape: (batch_size, seq_len, hidden_dim)
            """
            if current_sample_id is not None:
                # Store activation (detach and move to CPU to save GPU memory)
                if current_sample_id not in activations:
                    activations[current_sample_id] = {}
                
                # Output is the residual stream at this point
                # Shape: (1, seq_len, hidden_dim) for batch_size=1
                activations[current_sample_id][layer] = output[0].detach().cpu()
        
        # Hook into layer output (residual stream)
        layer_module = model.model.layers[layer_idx]
        handle = layer_module.register_forward_hook(hook_fn)
        handles.append(handle)
    
    print(f"Registered hooks for layers: {target_layers}")
    
    # Run forward passes
    print(f"\nCollecting activations for {len(samples)} samples...")
    
    for sample in tqdm(samples, desc="Activation collection"):
        current_sample_id = sample['sample_id']
        
        # Load audio
        audio_path = sample['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        # Prepare inputs
        question = sample['question']
        choices = sample['choices']
        
        choices_text = '\n'.join([f"{chr(65+i)}. {choice}" 
                                  for i, choice in enumerate(choices)])
        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        
        inputs = processor(
            text=[prompt],
            audios=[audio],
            return_tensors="pt",
            sampling_rate=sr
        ).to(model.device)
        
        # Forward pass (triggers hooks)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    print(f"  Collected activations for {len(activations)} samples")
    print(f"  Layers: {target_layers}")
    print(f"  Activation shape example: {list(activations.values())[0][target_layers[0]].shape}")
    
    return activations


# Target layers (every 8 layers for computational efficiency)
target_layers = [8, 16, 24, 31]

# Collect from MMAU extraction set (4,000 samples)
print("\n=== Collecting MMAU Activations ===")
mmau_activations = collect_residual_activations(
    model, processor, mmau_extraction_results, target_layers
)

# Collect from MMAR extraction set (400 samples)  
print("\n=== Collecting MMAR Activations ===")
mmar_activations = collect_residual_activations(
    model, processor, mmar_extraction_results, target_layers
)

# Save activations (WARNING: Large files!)
torch.save(mmau_activations, 'mmau_activations.pt')
torch.save(mmar_activations, 'mmar_activations.pt')

print("\nActivations saved")