# Download and convert model to MLX format (with quantization)
python -m mlx_lm.convert \
  --hf-path meta-llama/Meta-Llama-3-8B-Instruct \
  --mlx-path models/llama3-mlx \
  -q

# Fine-tune with LoRA for Llama-3-8B-Instruct
python -m mlx_lm.lora \
  --model models/llama3-mlx \
  --data data \
  --train \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --adapter-path adapters \
  --save-every 100


# Fine-tune directly (no conversion needed) for Gemma-3-1B-IT
TOKENIZERS_PARALLELISM=false python -m mlx_lm.lora \
  --model google/gemma-3-1b-it \
  --train \
  --iters 200 \
  --data data \
  --adapter-path ./checkpoints \
  --save-every 100 \
  --batch-size 1 \
  --grad-checkpoint