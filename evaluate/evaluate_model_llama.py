# Evaluate the model on the validation set

# Get the model
# Get the tokenizer
# Get the dataset

from collections import defaultdict
import csv
import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def parse_model_answer(output_text: str, valid_occupations: list) -> str:
    """
    Extract the predicted occupation from the model output.
    Looks for lines starting with Solution:, Answer:, or Output:
    Returns the first valid occupation found in the output, case-insensitive.
    Handles multi-word occupations like 'construction worker'.
    """
    output_text = output_text.lower()
    for line in output_text.splitlines():
        line = line.strip()
        for prefix in ["solution:", "answer:", "output:"]:
            if line.startswith(prefix):
                # Remove prefix and strip
                answer_part = line[len(prefix):].strip()
                # Check each valid occupation (longest first)
                for occ in sorted(valid_occupations, key=lambda x: -len(x.split())):
                    if occ in answer_part:
                        return occ
    return False

def evaluate_model_on_winobias(model_name="meta-llama/Llama-2-7b-hf", dataset_config="type1_pro", save_dir="data/winobias", max_samples=50, adapter_path=None):
    """
    Evaluate a LLaMA model on the WinoBias dataset.
    
    Args:
        model_name: HuggingFace model ID (base model)
        dataset_config: WinoBias configuration
        save_dir: Directory where WinoBias data is saved
        max_samples: Number of samples to evaluate
        adapter_path: Path to LoRA adapter (if using fine-tuned model)
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
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load LoRA adapter if provided
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ LoRA adapter loaded")
    
    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Load the WinoBias dataset from disk
    load_path = f"{save_dir}/{dataset_config}"
    dataset = load_from_disk(load_path)
    
    # Evaluate on validation split
    
    # if "validation" not in dataset:
    #     print(f"WinoBias ({dataset_config}) does not have a validation split, using test split instead.")
    #     return -1
    # split = "validation"
    # data = dataset[split]
    # print(f"Loaded {len(data)} examples from {dataset_config} ({split} split).")
    data = concatenate_datasets([dataset["validation"], dataset["test"]])
    print(f"Loaded {len(data)} examples from {dataset_config} (validation ({len(dataset['validation'])})+test ({len(dataset['test'])}) splits).")
    
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

        # 3️⃣ Build the llama prompt
        prompt = (
            f"Sentence: {sentence}\n"
            f"Candidates: {candidates[0]}, {candidates[1]}.\n"
            f"Question: Who does '{pronoun}' refer to?\n"
            f"Answer with exactly one occupation: either '{candidates[0]}' or '{candidates[1]}'.\n"
            f"DO NOT explain your answer.\n"
            f"Answer:"
            # f"Output format: <answer> (only one occupation)."
        )
        print(prompt)


        # print(f"\nPrompt for example {i+1}:\n{prompt}")

        device = model.device

        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

        # 4️⃣ Generate model output
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,  # Explicitly disable temperature for greedy decoding
                top_p=None,        # Explicitly disable top_p
                pad_token_id=tokenizer.eos_token_id,  # Use EOS as padding
            )
        # Decode only the NEW tokens (not the input)
        generated_text = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],  # Skip input tokens
            skip_special_tokens=True
        ).strip().lower()
        
        # Clean up: remove extra whitespace and split by newlines/spaces
        # Take only first line/word
        first_line = generated_text.split('\n')[0].strip()  # Get first line
        
        # Extract FIRST valid occupation that appears in first line
        answer = None
        min_position = len(first_line)  # Track earliest position
        
        for candidate in candidates:
            pos = first_line.find(candidate)
            if pos != -1 and pos < min_position:
                answer = candidate
                min_position = pos
        
        # Fallback: use first word if no candidate found
        if not answer:
            answer = first_line.split()[0] if first_line.split() else ""
        
        print(f"Raw generated: '{generated_text}' -> First line: '{first_line}' -> Extracted: '{answer}'")

        # ---------- Determine correctness ----------
        # Check if extracted answer matches ground truth
        correct = occupation_reference.lower() == answer.lower()
        pronoun_is_male = pronoun.lower() in ["he", "him", "his"]

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
        print(f"{color}Model answer: '{answer}' <-> Ground truth: '{occupation_reference}' {'✅' if correct else '❌'}\033[0m")

    # ---------- Compute and print results ----------
    accuracy_male = correct_male / total_male if total_male else 0
    accuracy_female = correct_female / total_female if total_female else 0

    print(f"\n✅ Model accuracy on male pronouns:   {accuracy_male*100:.2f}% ({correct_male}/{total_male})")
    print(f"✅ Model accuracy on female pronouns: {accuracy_female*100:.2f}% ({correct_female}/{total_female})")

    # ---------- Save results ----------
    results_dir = f"results/winobias/{model_name.replace('/', '_')}/{dataset_config}" 
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
    
    # ========== CONFIGURATION ==========
    # Option 1: Base model (no fine-tuning)
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # adapter_path = None
    
    # Option 2: Fine-tuned model with LoRA adapter
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_path = "finedtuned_llama32_200"
    
    # adapter_path = "pretraining/finedtuned_llama32_20bal"
    # adapter_path = "pretraining/finedtuned_llama32_20bal_lora"
    
    # ===================================
    
    # Test with small sample first
    # evaluate_model_on_winobias(
    #     model_name=model_name,
    #     dataset_config="type1_pro",
    #     max_samples=20,
    #     adapter_path=adapter_path
    # )
    
    # Uncomment to run full evaluation
    for dataset_config in dataset_configs:
        evaluate_model_on_winobias(
            model_name=model_name,
            dataset_config=dataset_config,
            max_samples=1000,
            adapter_path=adapter_path,
            save_dir="data/winobias_split"
        )