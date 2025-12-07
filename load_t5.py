import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------------------------
# 1. Select device: use MPS (Apple GPU) if available, otherwise CPU
# --------------------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------
# 2. Load the T5-base model and tokenizer
# --------------------------------------------------------------------
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# --------------------------------------------------------------------
# 3. Test inference: a simple example
# --------------------------------------------------------------------
input_text = "Translate English to French: The cat is on the mat."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    output_tokens = model.generate(**inputs, max_length=40)

# Decode the generated text
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print(f"\nInput:  {input_text}")
print(f"Output: {output_text}")