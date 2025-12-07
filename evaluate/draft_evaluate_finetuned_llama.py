# load the final model llama 3.2 that is in final_model

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
model_name = "/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/5ano/1semestre/KL/KL_src/pretraining/final_model"

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# 2. Load your LoRA adapter on top
model = PeftModel.from_pretrained(model, model_name)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Test the model with a sample input
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:", generated_text)    

