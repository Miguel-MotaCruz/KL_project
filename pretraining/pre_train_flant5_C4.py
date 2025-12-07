import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
import gc

# ========== CONFIGURATION ==========
MODEL_NAME = "google/flan-t5-base"  # Options: google/flan-t5-base, google/flan-t5-large, google/flan-t5-xl

# Training data
TRAIN_DATA_PATH = "pretraining/data_training/train_5bal.csv"  # Your knowledge graph triples
C4_DATA_PATH = "pretraining/data_training/c4_1000_news.csv"  # C4 pretraining data
VALIDATION_DATA_PATH = "data/winobias_split/winobias_finetune_validation.csv"  # WinoBias validation

# LoRA parameters
USE_LORA = True  # Set to False for full fine-tuning
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training parameters
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4×4 = 16
NUM_EPOCHS = 3
LEARNING_RATE = 5e-4 if USE_LORA else 5e-5  # Higher for LoRA
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128



# Output
OUTPUT_DIR = "./pretraining/finetuned_news1000c4_flant5_base_5bal_lora" if USE_LORA else "./pretraining/finetuned_news1000c4_flant5_base_5bal_full"  # Change this for different runs
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
    'validation': VALIDATION_DATA_PATH,
    'c4': C4_DATA_PATH
})

# Shuffle datasets
dataset['train'] = dataset['train'].shuffle(seed=42)
dataset['validation'] = dataset['validation'].shuffle(seed=42)
dataset['c4'] = dataset['c4'].shuffle(seed=42).select(range(min(len(dataset['train'])//8, len(dataset['c4']))))

# dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(200, len(dataset['train']))))
# dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(min(100, len(dataset['validation']))))


print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print(f"C4 examples: {len(dataset['c4'])}")

# join train with c4 shuffled
dataset['train'] = concatenate_datasets([dataset['train'], dataset['c4']])
dataset['train'] = dataset['train'].shuffle(seed=42)
print(f"Combined Train examples (train + c4): {len(dataset['train'])}")

#print 10 examples from train dataset
for i in range(10):
    print(f"\nExample {i+1}:")
    print("Prompt:", dataset['train'][i]['prompt'])
    print("Completion:", dataset['train'][i]['completion'])

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

for ex in dataset['train'][:10]:
    print("INPUT:", ex["input_str"])
    print("TARGET:", ex["target_str"])

print("\nStarting training...")
trainer.train()

# 7) Save model
print(f"\nSaving model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 Training complete! Model saved to {OUTPUT_DIR}\n")

