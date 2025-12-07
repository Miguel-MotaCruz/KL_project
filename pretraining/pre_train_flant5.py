import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import gc

# ========== CONFIGURATION ==========
MODEL_NAME = "google/flan-t5-base"  # Options: google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl

PROPORTION = "IMBALANCED"  # Options: BALANCED, IMBALANCED
if PROPORTION == "BALANCED":
    balanced_number = None
else:
    imbalanced_number = 200

# Training data
if PROPORTION == "BALANCED":
    TRAIN_DATA_PATH = f"pretraining/data_training/train_{balanced_number}bal.csv"  # Your knowledge graph triples
else:
    TRAIN_DATA_PATH = f"pretraining/data_training/train_{imbalanced_number}imbal.csv"  # Your knowledge graph triples
VALIDATION_DATA_PATH = "data/winobias_split/winobias_finetune_validation.csv"  # WinoBias validation

# LoRA parameters
USE_LORA = True  # Set to False for full fine-tuning
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
    OUTPUT_DIR = f"./pretraining/finetuned_flant5_base_{balanced_number}bal_lora_{learning_rate_str}" if USE_LORA else f"./pretraining/finetuned_flant5_base_{balanced_number}bal_full_{learning_rate_str}"  # Change this for different runs
else:
    OUTPUT_DIR = f"./pretraining/finetuned_flant5_base_{imbalanced_number}imbal_lora_{learning_rate_str}" if USE_LORA else f"./pretraining/finetuned_flant5_base_{imbalanced_number}imbal_full_{learning_rate_str}"  # Change this for different runs
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
        target_modules=["q", "v"],  # T5 uses "q" and "v" for attention layers
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
    'train': TRAIN_DATA_PATH,
    'validation': VALIDATION_DATA_PATH
})

# Shuffle train dataset by pairs (every 2 rows belong to same person)
import random
train_df = dataset['train'].to_pandas()
# Group into pairs
pairs = [(i, i+1) for i in range(0, len(train_df), 2)]
# Shuffle pairs
random.seed(42)
random.shuffle(pairs)
# Reconstruct dataset maintaining pair order
shuffled_indices = [idx for pair in pairs for idx in pair]
train_df_shuffled = train_df.iloc[shuffled_indices].reset_index(drop=True)
from datasets import Dataset
dataset['train'] = Dataset.from_pandas(train_df_shuffled)

# Shuffle validation normally
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
    eval_steps=40, #300
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

