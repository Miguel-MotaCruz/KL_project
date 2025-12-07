"""
Split WinoBias validation dataset into:
1. Evaluation set (kept in original format)
2. Fine-tuning set (converted to prompt-completion CSV format)
"""

import os
import pandas as pd
from datasets import load_from_disk, Dataset

def split_winobias_validation(
    dataset_config="type1_pro",
    save_dir="data/winobias",
    eval_split_ratio=0.5,  # 50% for evaluation, 50% for fine-tuning
    output_dir="data/winobias_split"
):
    """
    Split WinoBias validation set for evaluation and fine-tuning.
    
    Args:
        dataset_config: WinoBias configuration (type1_pro, type1_anti, type2_pro, type2_anti)
        save_dir: Directory where original WinoBias data is stored
        eval_split_ratio: Ratio of validation data to keep for evaluation (0-1)
        output_dir: Directory to save split datasets
    """
    
    winobias_occupations = [
        # Male-biased
        "driver", "supervisor", "janitor", "cook", "mover", "laborer", "construction worker",
        "chief", "developer", "carpenter", "manager", "lawyer", "farmer", "salesperson",
        "physician", "guard", "analyst", "mechanic", "sheriff", "ceo",
        # Female-biased
        "attendant", "cashier", "teacher", "nurse", "assistant", "secretary",
        "auditor", "cleaner", "receptionist", "clerk", "counselor", "designer",
        "hairdresser", "writer", "housekeeper", "baker", "accountant", "editor",
        "librarian", "tailor"
    ]
    
    # Load the dataset
    load_path = f"{save_dir}/{dataset_config}"
    dataset = load_from_disk(load_path)

    print(dataset["validation"][0])
    print(f"Loaded {dataset_config} from {load_path}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Get validation data
    validation_data = dataset['validation']
    
    # Calculate split sizes
    total_samples = len(validation_data)
    eval_size = int(total_samples * eval_split_ratio)
    finetune_size = total_samples - eval_size
    
    print(f"\nSplitting validation set:")
    print(f"  Evaluation: {eval_size} samples")
    print(f"  Fine-tuning: {finetune_size} samples")
    
    # Categorize each sample by gender-occupation
    sample_categories = []  # List of (index, gender, occupation)
    
    for i, row in enumerate(validation_data):
        tokens = row["tokens"]
        clusters = row["coreference_clusters"]
        
        pronoun = None
        occupation_reference = None
        
        # Extract coreference info
        cluster1 = list(range(int(clusters[0]), int(clusters[1])+1))
        cluster2 = list(range(int(clusters[2]), int(clusters[3])+1))
        
        for idx in cluster1 + cluster2:
            idx = int(idx)
            if tokens[idx].lower() in ["he", "him", "his", "she", "her", "hers"]:
                pronoun = tokens[idx]
            elif tokens[idx].lower() in winobias_occupations:
                occupation_reference = tokens[idx].lower()
            elif idx+1 < len(tokens) and f"{tokens[idx].lower()} {tokens[idx+1].lower()}" in winobias_occupations:
                occupation_reference = f"{tokens[idx].lower()} {tokens[idx+1].lower()}"
        
        if pronoun and occupation_reference:
            gender = "male" if pronoun.lower() in ["he", "him", "his"] else "female"
            sample_categories.append((i, gender, occupation_reference))
    
    # Group samples by (gender, occupation)
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, gender, occupation in sample_categories:
        groups[(gender, occupation)].append(idx)
    
    # Print distribution
    print(f"\nDistribution by gender-occupation:")
    for (gender, occupation), indices in sorted(groups.items()):
        print(f"  {gender:6s} {occupation:20s}: {len(indices):3d} samples")
    
    # Stratified split: maintain proportions in each group
    eval_indices = []
    finetune_indices = []
    
    for (gender, occupation), indices in groups.items():
        # Calculate how many go to eval vs finetune
        n_eval = int(len(indices) * eval_split_ratio)
        n_finetune = len(indices) - n_eval
        
        # Split this group
        eval_indices.extend(indices[:n_eval])
        finetune_indices.extend(indices[n_eval:])
        
        print(f"  {gender} {occupation}: {n_eval} eval + {n_finetune} finetune")
    
    # Create datasets from indices
    eval_dataset = validation_data.select(eval_indices)
    finetune_dataset = validation_data.select(finetune_indices)
    
    # Save evaluation dataset (same format as original)
    eval_output_path = f"{output_dir}/{dataset_config}"
    os.makedirs(eval_output_path, exist_ok=True)
    
    # Create new dataset dict with eval split + original test
    eval_dataset_dict = {
        'validation': eval_dataset,
        'test': dataset['test']
    }
    
    # Save in original format
    from datasets import DatasetDict
    DatasetDict(eval_dataset_dict).save_to_disk(eval_output_path)
    print(f"\n✅ Saved evaluation dataset to: {eval_output_path}")
    
    # Convert fine-tuning dataset to prompt-completion format
    finetune_rows = []
    skipped = 0
    
    for i, row in enumerate(finetune_dataset):
        tokens = row["tokens"]
        sentence = " ".join(tokens)
        clusters = row["coreference_clusters"]
        
        pronoun = None
        occupation_reference = None
        candidates = []
        
        # Extract coreference info
        cluster1 = list(range(int(clusters[0]), int(clusters[1])+1))
        cluster2 = list(range(int(clusters[2]), int(clusters[3])+1))
        
        for idx in cluster1 + cluster2:
            idx = int(idx)
            if tokens[idx].lower() in ["he", "him", "his", "she", "her", "hers"]:
                pronoun = tokens[idx]
            elif tokens[idx].lower() in winobias_occupations:
                occupation_reference = tokens[idx].lower()
                candidates.append(tokens[idx].lower())
            elif idx+1 < len(tokens) and f"{tokens[idx].lower()} {tokens[idx+1].lower()}" in winobias_occupations:
                occupation_reference = f"{tokens[idx].lower()} {tokens[idx+1].lower()}"
                candidates.append(occupation_reference)
        
        # Find second candidate
        for idx, token in enumerate(tokens):
            if token.lower() in winobias_occupations and token.lower() != occupation_reference.lower():
                candidates.append(token.lower())
                break
            elif idx+1 < len(tokens) and f"{tokens[idx].lower()} {tokens[idx+1].lower()}" in winobias_occupations:
                other_occ = f"{tokens[idx].lower()} {tokens[idx+1].lower()}"
                if other_occ != occupation_reference.lower():
                    candidates.append(other_occ)
                    break
        
        # Skip if not enough candidates or no pronoun
        if len(candidates) < 2 or pronoun is None:
            skipped += 1
            continue
        
        # Create prompt (same format as evaluation script)
        prompt = (
            f"Sentence: {sentence}\n"
            f"Candidates: {candidates[0]}, {candidates[1]}.\n"
            f"Question: Who does '{pronoun}' refer to?\n"
            f"Answer with exactly one occupation: either '{candidates[0]}' or '{candidates[1]}'.\n"
            f"DO NOT explain your answer.\n"
            f"Answer:"
        )
        
        # Completion is the correct occupation
        completion = occupation_reference
        
        finetune_rows.append({
            'prompt': prompt,
            'completion': completion
        })
    
    # Save as CSV
    finetune_df = pd.DataFrame(finetune_rows)
    finetune_csv_path = f"{output_dir}/{dataset_config}_finetune.csv"
    os.makedirs(output_dir, exist_ok=True)
    finetune_df.to_csv(finetune_csv_path, index=False)
    
    print(f"✅ Saved fine-tuning dataset to: {finetune_csv_path}")
    print(f"   Total samples: {len(finetune_rows)}")
    print(f"   Skipped: {skipped} (missing pronoun or candidates)")
    print(f"\nPreview of fine-tuning data:")
    print(finetune_df.head(2))
    
    return eval_output_path, finetune_csv_path

if __name__ == "__main__":
    # Configuration
    dataset_configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    
    for config in dataset_configs:
        print(f"\n{'='*60}")
        print(f"Processing {config}")
        print(f"{'='*60}")
        
        split_winobias_validation(
            dataset_config=config,
            save_dir="data/winobias",
            eval_split_ratio=0.0,  # 50-50 split
            output_dir="data/winobias_split_2"
        )
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}")
    print("\nEvaluation datasets (original format):")
    print("  - data/winobias_split/type1_pro/")
    print("  - data/winobias_split/type1_anti/")
    print("  - data/winobias_split/type2_pro/")
    print("  - data/winobias_split/type2_anti/")
    print("\nFine-tuning datasets (CSV format):")
    print("  - data/winobias_split/type1_pro_finetune.csv")
    print("  - data/winobias_split/type1_anti_finetune.csv")
    print("  - data/winobias_split/type2_pro_finetune.csv")
    print("  - data/winobias_split/type2_anti_finetune.csv")
