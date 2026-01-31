def comprehensive_evaluation(
    model, processor, test_set, steering_vectors, alpha,
    dataset_name="Test"
):
    """
    Comprehensive evaluation with breakdown by categories
    """
    
    # Apply steering
    model, handles = apply_steering(model, steering_vectors, alpha)
    
    # Results containers
    all_results = []
    
    # Category breakdowns
    by_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_modality = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print(f"\n=== Final Evaluation: {dataset_name} (alpha={alpha}) ===")
    
    for sample in tqdm(test_set, desc="Evaluation"):
        # Run inference (same as before)
        audio_path = sample['audio_path']
        audio, sr = torchaudio.load(audio_path)
        
        question = sample['question']
        choices = sample.get('choices', sample.get('options', []))
        
        choices_text = '\n'.join([f"{chr(65+i)}. {choice}" 
                                  for i, choice in enumerate(choices)])
        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        
        inputs = processor(
            text=[prompt],
            audios=[audio],
            return_tensors="pt",
            sampling_rate=sr
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        pred_letter = prediction[0] if prediction else 'A'
        
        ground_truth = sample.get('answer', sample.get('ground_truth'))
        if ground_truth in choices:
            gt_letter = chr(65 + choices.index(ground_truth))
        else:
            gt_letter = ground_truth[0] if ground_truth else 'A'
        
        is_correct = (pred_letter == gt_letter)
        
        # Store result
        result = {
            'sample_id': sample.get('id', sample.get('sample_id')),
            'prediction': pred_letter,
            'ground_truth': gt_letter,
            'is_correct': is_correct,
            'difficulty': sample.get('difficulty', 'unknown'),
            'modality': sample.get('modality', sample.get('domain', 'unknown')),
            'category': sample.get('category', 'unknown')
        }
        all_results.append(result)
        
        # Update category counts
        difficulty = result['difficulty']
        modality = result['modality']
        category = result['category']
        
        by_difficulty[difficulty]['total'] += 1
        by_modality[modality]['total'] += 1
        by_category[category]['total'] += 1
        
        if is_correct:
            by_difficulty[difficulty]['correct'] += 1
            by_modality[modality]['correct'] += 1
            by_category[category]['correct'] += 1
    
    # Remove steering
    remove_steering(handles)
    
    # Print comprehensive results
    print(f"\n{'='*60}")
    print(f"{dataset_name} Final Results (alpha={alpha})")
    print(f"{'='*60}")
    
    # Overall
    total_correct = sum(r['is_correct'] for r in all_results)
    total = len(all_results)
    overall_acc = total_correct / total * 100
    
    print(f"\nOverall:")
    print(f"  Accuracy: {overall_acc:.2f}% ({total_correct}/{total})")
    
    # By difficulty
    if any(by_difficulty.values()):
        print(f"\nBy Difficulty:")
        for diff in sorted(by_difficulty.keys()):
            stats = by_difficulty[diff]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {diff:10s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})")
    
    # By modality
    if any(by_modality.values()):
        print(f"\nBy Modality:")
        for mod in sorted(by_modality.keys()):
            stats = by_modality[mod]
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {mod:15s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})")
    
    # By category (for MMAR)
    if any(by_category.values()) and dataset_name == "MMAR":
        print(f"\nBy Reasoning Layer:")
        for cat in sorted(by_category.keys()):
            stats = by_category[cat]
            if stats['total'] > 0 and cat != 'unknown':
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {cat:20s}: {acc:5.2f}% ({stats['correct']}/{stats['total']})")
    
    return all_results, overall_acc


# Final evaluation on MMAU
print("\n" + "="*80)
print("MMAU TEST SET - FINAL EVALUATION")
print("="*80)

mmau_final_results, mmau_final_acc = comprehensive_evaluation(
    model, processor, mmau_test, mmau_steering_vectors,
    alpha=best_mmau_alpha, dataset_name="MMAU"
)

# Final evaluation on MMAR
print("\n" + "="*80)
print("MMAR TEST SET - FINAL EVALUATION")
print("="*80)

mmar_final_results, mmar_final_acc = comprehensive_evaluation(
    model, processor, mmar_test, mmar_steering_vectors,
    alpha=best_mmar_alpha, dataset_name="MMAR"
)