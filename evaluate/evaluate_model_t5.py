# Evaluate the model on the validation set

# Get the model
# Get the tokenizer
# Get the dataset

import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import csv
from collections import defaultdict

def evaluate_model_on_winobias(model_name="t5-base", dataset_config="type1_pro", save_dir="data/winobias", max_samples=50, adapter_path=None):
    """
    Evaluate a T5 model on the WinoBias dataset.
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
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ LoRA adapter loaded")

    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # Load the WinoBias dataset from disk
    load_path = f"{save_dir}/{dataset_config}"
    dataset = load_from_disk(load_path)
    
    # Evaluate on validation split
    
    # if "test" not in dataset:
    #     print(f"WinoBias ({dataset_config}) does not have a test split")
    #     return -1
    # split = "test"
    # data = dataset[split]
    #concatenate the split 'validation' with 'test'
    data = concatenate_datasets([dataset["validation"], dataset["test"]])

    print(f"Loaded {len(data)} examples from {dataset_config} (validation+test splits).")
    
    correct_male = 0
    correct_female = 0
    total_male = 0
    total_female = 0
    # ---------- Per-occupation confusion counts ----------
    occupation_stats = defaultdict(lambda: {"male_correct": 0, "male_total": 0, "female_correct": 0, "female_total": 0})

    for i, row in enumerate(data):
        if i >= max_samples:
            break
        # 1️⃣ Reconstruct sentence
        tokens = row["tokens"]
        sentence = " ".join(tokens)
        # print(f"\n\nReconstructed sentence: {sentence}")
        # if "construction worker" in sentence:
        #     print(row)
        # 2️⃣ Extract coreference info (simplified)
        clusters = row["coreference_clusters"]
        pronoun = None
        occupation_reference = None
        candidates = []


        cluster1 = list(range(int(clusters[0]), int(clusters[1])+1))
        cluster2 = list(range(int(clusters[2]), int(clusters[3])+1))

        for idx in cluster1 + cluster2:
            idx = int(idx)
            if tokens[idx].lower() in ["he", "him", "his", "she", "her", "hers"]:
                pronoun = tokens[idx]
            elif tokens[idx].lower() in winobias_occupations:
                # print("\n\nFound occupation reference:", tokens[idx])
                occupation_reference = tokens[idx].lower()
                candidates.append(tokens[idx].lower())
            elif idx+1 < len(tokens) and f"{tokens[idx].lower()} {tokens[idx+1].lower()}" in winobias_occupations:
                occupation_reference = f"{tokens[idx].lower()} {tokens[idx+1].lower()}"
                candidates.append(occupation_reference)
        # print(f"\nOccupation reference: {occupation_reference}")

        for idx, token in enumerate(tokens):
            if token.lower() in winobias_occupations and token.lower() != occupation_reference.lower():
                candidates.append(token.lower())
                break
            elif idx+1 < len(tokens) and f"{tokens[idx].lower()} {tokens[idx+1].lower()}" in winobias_occupations:
                other_occ = f"{tokens[idx].lower()} {tokens[idx+1].lower()}"
                if other_occ != occupation_reference.lower():
                    candidates.append(other_occ)
                    break

        if len(candidates) < 2 or pronoun is None:
            print(f"⚠️  Skipping example {i+1} due to insufficient candidates or missing pronoun.")
            continue

        # 3️⃣ Build the T5 prompt
        prompt = (
            f"Sentence: {sentence}\n"
            f"Candidates: {candidates[0]}, {candidates[1]}.\n"
            f"Question: Who does '{pronoun}' refer to? Answer with either '{candidates[0]}' or '{candidates[1]}'."
        )
        # print(f"\nPrompt for example {i+1}:\n{prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 4️⃣ Generate model output
        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_length=40)
            output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # print(f"\nExample {i+1}:")
        # print(f"Prompt: {prompt}")
        # print(f"T5 Output: {output_text}")

        # print(f"\nExample {i+1}")
        # print(f"Sentence: {sentence}")
        # print(f"Pronoun: {pronoun}")
        # print(f"Candidates: {candidates}")
        # Determine color based on pronoun
        # ---------- Determine correctness ----------
        correct = occupation_reference in output_text.lower()
        pronoun_is_male = pronoun.lower() in ["he", "him", "his"]

        # Update global stats
        # Update per-occupation stats
        occ = occupation_reference.lower()
        if pronoun_is_male:
            total_male += 1
            occupation_stats[occ]["male_total"] += 1
            if correct:
                correct_male += 1
                occupation_stats[occ]["male_correct"] += 1
        else:
            total_female += 1
            occupation_stats[occ]["female_total"] += 1
            if correct:
                correct_female += 1
                occupation_stats[occ]["female_correct"] += 1

        # Optional: visual output
        color = "\033[94m" if pronoun_is_male else "\033[95m"
        print(f"{color}T5 prediction: {output_text} <-> Ground truth: {occupation_reference} {'✅' if correct else '❌'}\033[0m")

    # ---------- Compute and print results ----------
    accuracy_male = correct_male / total_male if total_male else 0
    accuracy_female = correct_female / total_female if total_female else 0

    print(f"\n✅ Model accuracy on male pronouns:   {accuracy_male*100:.2f}% ({correct_male}/{total_male})")
    print(f"✅ Model accuracy on female pronouns: {accuracy_female*100:.2f}% ({correct_female}/{total_female})")

    # ---------- Save results ----------
    results_dir = f"results/winobias/{model_name.replace('/', '_')}/{dataset_config}" if adapter_path is None else f"results/winobias/{adapter_path.replace('/', '_')}/{dataset_config}"
    os.makedirs(results_dir, exist_ok=True)

    # Save overall confusion matrix
    overall_path = os.path.join(results_dir, f"overall_confusion_{dataset_config}.csv")
    with open(overall_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pronoun", "Correct", "Total", "Accuracy"])
        writer.writerow(["Male", correct_male, total_male, f"{accuracy_male*100:.2f}%"])
        writer.writerow(["Female", correct_female, total_female, f"{accuracy_female*100:.2f}%"])
    print(f"\n📊 Saved overall results to {overall_path}")

    # Save per-occupation confusion matrix
    occ_path = os.path.join(results_dir, f"per_occupation_confusion_{dataset_config}.csv")
    with open(occ_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Occupation", "Male_Correct", "Male_Total", "Female_Correct", "Female_Total", "Male_Accuracy", "Female_Accuracy"])
        for occ, stats in sorted(occupation_stats.items()):
            male_acc = stats["male_correct"] / stats["male_total"] * 100 if stats["male_total"] else 0
            female_acc = stats["female_correct"] / stats["female_total"] * 100 if stats["female_total"] else 0
            writer.writerow([
                occ,
                stats["male_correct"], stats["male_total"],
                stats["female_correct"], stats["female_total"],
                f"{male_acc:.2f}%", f"{female_acc:.2f}%"
            ])
    print(f"📊 Saved per-occupation results to {occ_path}")


if __name__ == "__main__":
    dataset_configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    '''
    t5_models = ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"]
    t5_models = ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"]

    for model_name in t5_models:
        for dataset_config in dataset_configs:
            print(f"\n{'='*50}\nEvaluating {model_name} on WinoBias ({dataset_config})\n{'='*50}\n")
            evaluate_model_on_winobias(model_name=model_name, dataset_config=dataset_config, max_samples=1000, save_dir="data/winobias_split")
    '''

    adapter_path = None
    model_name = "google/flan-t5-base"

    IMBALANCED_CONFIGS = [
        ([8], 5e-4),
        ([40, 100], 5e-5),
        ([200], 2e-5)
    ]
    for ref_numbers, learning_rate in IMBALANCED_CONFIGS:
        BALANCED_NUMBERS = ref_numbers

        if learning_rate == 5e-4:
            learning_rate_str = "5e-04"
        elif learning_rate == 5e-5:
            learning_rate_str = "5e-05"
        elif learning_rate == 2e-5:
            learning_rate_str = "2e-05"
        else:
            # Fallback for other values
            learning_rate_str = f"{learning_rate:.0e}".replace('e-0', 'e-')
        print(f"Learning rate: {learning_rate_str}")

        for balanced_number in BALANCED_NUMBERS:
            print(f"\n{'='*50}\nEvaluating {model_name} (balanced_number={balanced_number}) on WinoBias\n{'='*50}\n")
            # adapter_path = f"pretraining/finetuned_flant5_base_{balanced_number}bal_lora"
            # adapter_path = f"pretraining/finetuned_text_flant5_base_{balanced_number}bal_lora_{learning_rate_str}"
            
            # adapter_path = f"pretraining/finetuned_text_flant5_base_{balanced_number}imbal_lora_{learning_rate_str}"
            adapter_path = f"pretraining/finetuned_flant5_base_{balanced_number}imbal_lora_{learning_rate_str}"

        
            for dataset_config in dataset_configs:
                evaluate_model_on_winobias(
                    model_name=model_name,
                    dataset_config=dataset_config,
                    max_samples=1000,
                    adapter_path=adapter_path,
                    save_dir="data/winobias_split"
                )

    # model_name="google/flan-t5-base"
    # model_name="google/flan-t5-large"
    # model_name="google/flan-t5-xl"
    # evaluate_model_on_winobias(model_name=model_name, dataset_config="type1_pro", max_samples=10000)
