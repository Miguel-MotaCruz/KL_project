import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json
import gc

# Clear MPS cache from previous runs
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
gc.collect()

# Check MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps",
    attn_implementation="eager"  # Required for MPS
)

'''
How LoRA works:
Instead of updating all the weights in the model's attention layers, LoRA:

Freezes the original model weights completely
Adds small adapter matrices (rank decomposition) to specific layers
Only trains these small adapters (the 1.7M parameters)
'''

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# FIXED: Prepare dataset function for batched processing
def prepare_data(examples):
    # When batched=True, examples is a dict with lists as values
    # e.g., {'prompt': ['Q1', 'Q2'], 'completion': ['A1', 'A2']}
    texts = [f"<prompt>{prompt}</prompt><response>{completion}</response>" 
             for prompt, completion in zip(examples['prompt'], examples['completion'])]
    
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_tensors=None  # Return lists, not tensors
    )
    
    # Add labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Load your data
# dataset = load_dataset('json', data_files={
#     'train': 'data_training/train.jsonl',
#     'validation': 'data_training/valid.jsonl'
# })

dataset = load_dataset('csv', data_files={
    # 'train': 'pretraining/data_training/train_200.csv',
    # 'train': 'pretraining/data_training/train_20bal.csv',
    'train': 'pretraining/data_training/train_100bal.csv',
    'validation': 'data/winobias_split/winobias_finetune_validation.csv'
})

# Limit dataset size for quick testing with random sampling
# dataset['train'] = dataset['train'].shuffle(seed=42).select(range(min(500, len(dataset['train']))))
# dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(min(250, len(dataset['validation']))))

dataset['train'] = dataset['train'].shuffle(seed=42)
dataset['validation'] = dataset['validation'].shuffle(seed=42)

print(f"Train examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")

# Tokenize dataset
tokenized_dataset = dataset.map(
    prepare_data, 
    batched=True, 
    remove_columns=dataset['train'].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,  # Safe for M1 Pro 32GB with MPS
    gradient_accumulation_steps=4,  # Effective batch = 4×4 = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,  # Changed: fp16 doesn't work with MPS
    logging_steps=75,  # Increased for 13k dataset (was 10)
    save_steps=800,    # Adjusted: ~every epoch with 13k samples (was 100)
    eval_strategy="steps",
    eval_steps=300,    # Evaluate 2-3 times per epoch (was 50)
    save_total_limit=3,
    warmup_steps=100,  # Increased for larger dataset (was 10)
    report_to="none",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("\nStarting training...")
trainer.train()

# Save model
print("\nSaving model...")
modelsave_name = "finedtuned_llama32_20bal" 
model.save_pretrained(f"./pretraining/{modelsave_name}")
tokenizer.save_pretrained(f"./pretraining/{modelsave_name}")
print(f"Training complete! Model saved to ./pretraining/{modelsave_name}")