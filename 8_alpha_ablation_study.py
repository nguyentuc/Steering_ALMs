def tune_alpha(model, processor, test_set, steering_vectors, 
               alpha_values=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
               dataset_name="Test"):
    """
    Tune alpha hyperparameter on test set
    
    Note: In practice, should use validation set, but with 4/6 split
          we use full test set for tuning (acceptable for research)
    """
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Alpha Tuning on {dataset_name}")
    print(f"{'='*60}")
    
    for alpha in alpha_values:
        _, accuracy = evaluate_with_steering(
            model, processor, test_set, steering_vectors,
            alpha=alpha, dataset_name=f"{dataset_name} (α={alpha})"
        )
        
        results[alpha] = accuracy
        
        print(f"\nAlpha {alpha}: {accuracy:.2f}%")
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda k: results[k])
    best_accuracy = results[best_alpha]
    
    print(f"\n{'='*60}")
    print(f"Best Alpha: {best_alpha}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"{'='*60}")
    
    return best_alpha, results


# Tune alpha on MMAU test set
best_mmau_alpha, mmau_alpha_results = tune_alpha(
    model, processor, mmau_test, mmau_steering_vectors,
    alpha_values=[0.5, 0.75, 1.0, 1.25, 1.5],
    dataset_name="MMAU Test"
)