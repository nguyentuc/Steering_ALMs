import json
import random
from collections import defaultdict
import pickle

# Download and split MMAU datasets which will be used for steering vector extraction
# MMAU: 10,000 samples total

BASE_DIR = Path("/media/volume/h100_instance2/LALMs_Reasoning_Steering/MMAU")
AUDIO_BASE_DIR = BASE_DIR / "data/test-mini-audios/"
TEST_JSON = BASE_DIR / "mmau-test-mini.json"

def load_datasets():
    """Load MMAU and MMAR datasets"""
    
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")
    return data

def stratified_split(data, split_ratio=0.4, stratify_keys=['difficulty', 'modality']):
    """
    Perform stratified 4/6 split preserving distribution
    
    Args:
        data: List of samples
        split_ratio: 0.4 = 40% extraction, 60% test
        stratify_keys: Keys to stratify on
    
    Returns:
        extraction_set: 40% of data
        test_set: 60% of data
    """
    from sklearn.model_selection import train_test_split
    
    # Create stratification labels
    # Combine multiple keys into single stratification label
    stratify_labels = []
    for sample in data:
        label_parts = [sample.get(key, 'unknown') for key in stratify_keys]
        stratify_label = '_'.join(map(str, label_parts))
        stratify_labels.append(stratify_label)
    
    # Perform stratified split
    indices = list(range(len(data)))
    
    train_indices, test_indices = train_test_split(
        indices,
        test_size=(1 - split_ratio),  # 60% test
        stratify=stratify_labels,
        random_state=42
    )
    
    extraction_set = [data[i] for i in train_indices]
    test_set = [data[i] for i in test_indices]
    
    print(f"\nSplit Statistics:")
    print(f"Extraction set: {len(extraction_set)} samples ({len(extraction_set)/len(data)*100:.1f}%)")
    print(f"Test set: {len(test_set)} samples ({len(test_set)/len(data)*100:.1f}%)")
    
    # Verify stratification
    verify_stratification(extraction_set, test_set, stratify_keys)
    
    return extraction_set, test_set


def verify_stratification(extraction_set, test_set, keys):
    """Verify that distributions are preserved"""
    
    print("\n=== Stratification Verification ===")
    
    for key in keys:
        print(f"\n{key} distribution:")
        
        # Count in extraction set
        extraction_counts = defaultdict(int)
        for sample in extraction_set:
            extraction_counts[sample.get(key, 'unknown')] += 1
        
        # Count in test set
        test_counts = defaultdict(int)
        for sample in test_set:
            test_counts[sample.get(key, 'unknown')] += 1
        
        # Print distributions
        all_values = set(extraction_counts.keys()) | set(test_counts.keys())
        for value in sorted(all_values):
            ext_pct = extraction_counts[value] / len(extraction_set) * 100
            test_pct = test_counts[value] / len(test_set) * 100
            
            print(f"  {value:20s}: "
                  f"Extraction {ext_pct:5.1f}% | "
                  f"Test {test_pct:5.1f}% | "
                  f"Diff {abs(ext_pct - test_pct):5.2f}%")
            
# Load data
mmau_data = load_datasets()

# MMAU split (10,000 samples)
mmau_extraction, mmau_test = stratified_split(
    mmau_data,
    split_ratio=0.4,
    stratify_keys=['difficulty', 'domain']  # sound/music/speech
)
# Result: 4,000 extraction, 6,000 test

splits = {
    'mmau_extraction': mmau_extraction,
    'mmau_test': mmau_test
}

with open('dataset_splits.pkl', 'wb') as f:
    pickle.dump(splits, f)

print("\nSplits saved to dataset_splits.pkl")