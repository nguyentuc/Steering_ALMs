def separate_correct_incorrect(inference_results, activations):
    """
    Separate samples into correct and incorrect groups
    """
    
    correct_samples = []
    incorrect_samples = []
    
    for result in inference_results:
        sample_id = result['sample_id']
        
        # Skip if no activation (shouldn't happen but safety check)
        if sample_id not in activations:
            continue
        
        if result['is_correct']:
            correct_samples.append(sample_id)
        else:
            incorrect_samples.append(sample_id)
    
    print(f"Correct samples: {len(correct_samples)}")
    print(f"Incorrect samples: {len(incorrect_samples)}")
    
    return correct_samples, incorrect_samples


mmau_correct, mmau_incorrect = separate_correct_incorrect(
    mmau_extraction_results, mmau_activations
)


# Compute the mean different
def compute_mean_activations(activations, sample_ids, layers):
    """
    Compute mean activation for each layer
    
    Process:
    1. For each sample, average across sequence length
    2. Stack all sample activations
    3. Compute mean across samples
    
    Returns:
        mean_activations: Dict[layer] = tensor (hidden_dim,)
    """
    
    mean_activations = {}
    
    for layer in layers:
        layer_activations = []
        
        for sample_id in sample_ids:
            # Get activation for this sample at this layer
            # Shape: (seq_len, hidden_dim)
            activation = activations[sample_id][layer]
            
            # Average across sequence length
            # Shape: (hidden_dim,)
            activation_mean = activation.mean(dim=0)
            
            layer_activations.append(activation_mean)
        
        # Stack all samples: (num_samples, hidden_dim)
        layer_activations_stacked = torch.stack(layer_activations)
        
        # Compute mean across samples: (hidden_dim,)
        mean_activation = layer_activations_stacked.mean(dim=0)
        
        mean_activations[layer] = mean_activation
        
        print(f"Layer {layer}: "
              f"{len(sample_ids)} samples, "
              f"shape {mean_activation.shape}")
    
    return mean_activations


print("\n=== Computing Mean Activations ===")

# MMAU - Correct samples
print("\nMMU Correct:")
mmau_correct_means = compute_mean_activations(
    mmau_activations, mmau_correct, target_layers
)

# MMAU - Incorrect samples
print("\nMMU Incorrect:")
mmau_incorrect_means = compute_mean_activations(
    mmau_activations, mmau_incorrect, target_layers
)


## Extract the steering vector
def extract_steering_vectors(correct_means, incorrect_means, layers, normalize=True):
    """
    Extract steering vectors as mean difference
    
    Args:
        correct_means: Dict[layer] = mean activation for correct samples
        incorrect_means: Dict[layer] = mean activation for incorrect samples
        layers: List of layer indices
        normalize: Whether to L2-normalize vectors
    
    Returns:
        steering_vectors: Dict[layer] = steering vector (hidden_dim,)
    """
    
    steering_vectors = {}
    
    print("\n=== Extracting Steering Vectors ===")
    
    for layer in layers:
        # Mean difference: correct - incorrect
        steering_vec = correct_means[layer] - incorrect_means[layer]
        
        # Compute magnitude before normalization
        magnitude = steering_vec.norm().item()
        
        # Normalize (L2 norm = 1)
        if normalize:
            steering_vec = steering_vec / steering_vec.norm()
            normalized_magnitude = steering_vec.norm().item()
        else:
            normalized_magnitude = magnitude
        
        steering_vectors[layer] = steering_vec
        
        print(f"Layer {layer}:")
        print(f"  Original magnitude: {magnitude:.4f}")
        print(f"  Normalized magnitude: {normalized_magnitude:.4f}")
        print(f"  Shape: {steering_vec.shape}")
    
    return steering_vectors

# Extract MMAU steering vectors
mmau_steering_vectors = extract_steering_vectors(
    mmau_correct_means,
    mmau_incorrect_means,
    target_layers,
    normalize=True
)


# Analyze steering vector quality
def analyze_steering_vectors(steering_vectors, layers):
    """
    Analyze quality and properties of steering vectors
    """
    import torch.nn.functional as F
    
    print("\n=== Steering Vector Analysis ===")
    
    # 1. Cross-layer similarity
    print("\nCross-Layer Cosine Similarity:")
    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers[i+1:], start=i+1):
            vec_i = steering_vectors[layer_i]
            vec_j = steering_vectors[layer_j]
            
            similarity = F.cosine_similarity(
                vec_i.unsqueeze(0),
                vec_j.unsqueeze(0)
            ).item()
            
            print(f"  Layer {layer_i} ↔ Layer {layer_j}: {similarity:.4f}")
    
    # 2. Vector statistics
    print("\nVector Statistics:")
    for layer in layers:
        vec = steering_vectors[layer]
        
        print(f"\nLayer {layer}:")
        print(f"  Mean: {vec.mean().item():.6f}")
        print(f"  Std: {vec.std().item():.6f}")
        print(f"  Min: {vec.min().item():.6f}")
        print(f"  Max: {vec.max().item():.6f}")
        print(f"  L2 norm: {vec.norm().item():.6f}")
    
    return None

# Analyze MMAU vectors
print("\n" + "="*60)
print("MMAU Steering Vectors")
print("="*60)
analyze_steering_vectors(mmau_steering_vectors, target_layers)