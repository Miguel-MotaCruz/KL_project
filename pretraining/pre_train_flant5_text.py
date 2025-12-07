import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import gc
import csv
import pandas as pd
import random

# ========== CONFIGURATION ==========
MODEL_NAME = "google/flan-t5-base"  # Options: google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl


PROPORTION = "IMBALANCED"  # Options: BALANCED, IMBALANCED
if PROPORTION == "BALANCED":
    balanced_number = 20  
else:
    imbalanced_number = 200

# Training data
if PROPORTION == "BALANCED":
    TRAIN_DATA_PATH = f"pretraining/data_training/dataset_balanced{balanced_number}.csv"  # Your knowledge graph triples
else:
    TRAIN_DATA_PATH = f"pretraining/data_training/dataset_imbalanced{imbalanced_number}.csv"  # Your knowledge graph triples

VALIDATION_DATA_PATH = "data/winobias_split/winobias_finetune_validation.csv"  # WinoBias validation

# LoRA parameters
USE_LORA = True  # Set to False for full fine-tuning
# LORA_R = 8
# LORA_ALPHA = 16
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

# Training parameters
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4×4 = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-05 if USE_LORA else 2e-05  # Higher for LoRA
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Format LR to match pattern: 5e-04, 5e-05, 2e-05
if LEARNING_RATE == 5e-4:
    learning_rate_str = "5e-04"
elif LEARNING_RATE == 5e-5:
    learning_rate_str = "5e-05"
elif LEARNING_RATE == 2e-5:
    learning_rate_str = "2e-05"
else:
    # Fallback for other values
    learning_rate_str = f"{LEARNING_RATE:.0e}".replace('e-0', 'e-')
print(f"Learning rate: {learning_rate_str}")

# Output
if PROPORTION == "BALANCED":
    OUTPUT_DIR = f"./pretraining/finetuned_text_flant5_base_{balanced_number}bal_lora_{learning_rate_str}" if USE_LORA else f"./pretraining/finetuned_text_flant5_base_{balanced_number}bal_full_{learning_rate_str}"  # Change this for different runs
else:
    OUTPUT_DIR = f"./pretraining/finetuned_text_flant5_base_{imbalanced_number}imbal_lora_{learning_rate_str}" if USE_LORA else f"./pretraining/finetuned_text_flant5_base_{imbalanced_number}imbal_full_{learning_rate_str}"  # Change this for different runs
# ===================================

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

# ===================================



# Clear MPS cache from previous runs
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

# 1) Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")

print(f"Loading tokenizer & model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Apply LoRA if enabled
if USE_LORA:
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # target_modules=["q", "v"],  # T5 uses "q" and "v" for attention layers
        target_modules=[
            "SelfAttention.q",
            "SelfAttention.v",
            "EncDecAttention.q",
            "EncDecAttention.v"
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # Important: T5 is seq2seq, not causal LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

model.to(device)

# 2) Load datasets
print(f"\nLoading training data from: {TRAIN_DATA_PATH}")
print(f"Loading validation data from: {VALIDATION_DATA_PATH}")

dataset = load_dataset('csv', data_files={
    # 'train': TRAIN_DATA_PATH,
    'validation': VALIDATION_DATA_PATH
})
dataset['train'] = pd.DataFrame(columns=['prompt', 'completion'])

train_csv = pd.read_csv(TRAIN_DATA_PATH)


# ===== convert each pair of triples to text =====
import random

def triples_to_text2(triple_gender, triple_occ, all_occupations):
    """
    Convert two triples (gender + occupation) into a training example.
    Includes a random distractor occupation that does NOT appear in the sentence.

    Returns (input_text, target_text)

    triple_gender = (subject, "has gender", gender)
    triple_occ    = (subject, "has occupation", occupation)

    all_occupations = list of all possible occupation strings
    """

    subj, _, gender = triple_gender
    subj2, _, occupation = triple_occ

    assert subj == subj2, f"Triples must reference same subject: {subj} != {subj2}"

    # Normalize gender
    gender = gender.lower()
    is_male = gender in ["male", "man", "m"]
    is_female = not is_male

    # Pronouns
    pronoun = "he" if is_male else "she"
    pronoun_obj = "him" if is_male else "her"
    pronoun_poss = "his" if is_male else "her"
    pronoun_cap = pronoun.capitalize()

    # Pick a random *incorrect* occupation as distractor
    distractors = [o for o in all_occupations if o.lower() != occupation.lower()]
    random_occ = random.choice(distractors)

    # Build candidate list (random order!)
    candidates = [occupation, random_occ]
    random.shuffle(candidates)

    # Templates that include candidates
    templates = [

        # --- Template 1: simple statement + candidates ---
        (
            f"{subj} is {gender}. {pronoun_cap} works as a {occupation}. "
            f"Candidates: {candidates[0]}, {candidates[1]}. "
            f"Question: What job does {pronoun} have?",
            occupation
        ),

        # --- Template 2: reversed facts + candidates ---
        (
            f"{subj} works as a {occupation}. {pronoun_cap} is {gender}. "
            f"Candidates: {candidates[0]}, {candidates[1]}. "
            f"Which job is correct?",
            occupation
        ),

        # --- Template 3: short QA style ---
        (
            f"{subj} is {gender}. {pronoun_cap} has a profession. "
            f"Candidates: {candidates[0]}, {candidates[1]}. "
            f"Which profession is {pronoun_poss}?",
            occupation
        ),

        # --- Template 4: conversational ---
        (
            f"I know that {subj} is {gender} and works as a {occupation}. "
            f"Here are some options: {candidates[0]}, {candidates[1]}. "
            f"Which one is correct?",
            occupation
        ),

        # --- Template 5: factual + distractor awareness ---
        (
            f"Information: {subj} is {gender}. {pronoun_cap} works as a {occupation}. "
            f"Options: {candidates[0]} or {candidates[1]}. "
            f"Select the correct occupation.",
            occupation
        )
    ]

    return random.choice(templates)


def triples_to_text(triple_gender, triple_occ):
    """
    Convert two triples (gender + occupation) into a text prompt + target.
    Returns (input_text, target_text)
    
    triple_gender = (subject, "has gender", gender)
    triple_occ    = (subject, "has occupation", occupation)
    """

    subj, _, gender = triple_gender
    subj2, _, occupation = triple_occ

    assert subj == subj2, f"Triples must refer to the same person. {subj} != {subj2}"

    # normalize gender
    gender = gender.lower()
    is_male = gender in ["male", "man", "m"]
    is_female = gender in ["female", "woman", "f"]

    # correct pronouns
    pronoun = "he" if is_male else "she"
    pronoun_obj = "him" if is_male else "her"
    pronoun_poss = "his" if is_male else "her"
    pronoun_cap = pronoun.capitalize()

    templates = [

        # --- Template 1: simple fact ---
        (
            f"{subj} is {gender}. {pronoun_cap} works as a {occupation}.",
            occupation
        ),

        # --- Template 2: QA format (like evaluation style) ---
        (
            f"Context: {subj} is {gender}. {pronoun_cap} works as a {occupation}. "
            f"Question: What is {pronoun_poss} occupation?",
            occupation
        ),

        # --- Template 3: reverse fact (tests model reasoning) ---
        (
            f"{subj} works as a {occupation}. {pronoun_cap} is {gender}.",
            occupation
        ),

        # --- Template 4: “job question” style ---
        (
            f"{subj} is {gender}. What job does {pronoun} have?",
            occupation
        ),

        # --- Template 5: short description + question ---
        (
            f"{subj} is {gender} and has the profession of a {occupation}. "
            f"What is {pronoun_poss} profession?",
            occupation
        ),

        # --- Template 6: pronoun-only reasoning ---
        (
            f"{subj} is {gender}. {pronoun_cap} has a job. "
            f"Question: What job does {pronoun} have?",
            occupation
        ),

        # --- Template 7: direct QA, conversational tone ---
        (
            f"I know that {subj} is {gender}. Can you tell me what {pronoun} works as?",
            occupation
        ),

        # --- Template 8: fact consolidation ---
        (
            f"Here is some information: {subj} is {gender}. "
            f"{pronoun_cap} works professionally as a {occupation}.",
            occupation
        ),
    ]

    # random template selection
    return random.choice(templates)

# Build training data from triples
train_data = []
for i in range(0, len(train_csv), 2):
    triple1 = train_csv.iloc[i]
    triple2 = train_csv.iloc[i+1]
    prompt, completion = triples_to_text2(triple1, triple2, winobias_occupations)
    train_data.append({
        'prompt': prompt,
        'completion': completion
    })

# Convert to Dataset
dataset['train'] = Dataset.from_pandas(pd.DataFrame(train_data))

# save dataset to csv for verification
dataset['train'].to_csv(TRAIN_DATA_PATH.replace(".csv", "_textformat.csv"), index=False)

# Shuffle datasets
dataset['train'] = dataset['train'].shuffle(seed=42)
dataset['validation'] = dataset['validation'].shuffle(seed=42)

# dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(200, len(dataset['train']))))
# dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(min(100, len(dataset['validation']))))



print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")

# 3) Preprocessing function for T5 (encoder-decoder)
def preprocess(batch):
    """
    T5 expects:
    - input_ids: from the prompt (encoder input)
    - labels: from the completion (decoder target)
    """
    # Tokenize prompts (encoder input)
    model_inputs = tokenizer(
        batch["prompt"], 
        padding="max_length",
        truncation=True, 
        max_length=MAX_INPUT_LENGTH
    )
    
    # Tokenize completions (decoder target)
    labels = tokenizer(
        batch["completion"], 
        padding="max_length",
        truncation=True, 
        max_length=MAX_TARGET_LENGTH
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
print("\nTokenizing datasets...")
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

# 4) Data collator for seq2seq
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5) Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=torch.cuda.is_available(),  # FP16 only if CUDA available
    logging_steps=75,
    save_steps=800,
    eval_strategy="steps",
    eval_steps=120, #40, #300
    save_total_limit=3,
    warmup_steps=100,
    report_to="none",
)

# 6) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=collator,
)

print("\nStarting training...")
trainer.train()

# 7) Save model
print(f"\nSaving model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 Training complete! Model saved to {OUTPUT_DIR}\n")

