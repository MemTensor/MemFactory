#!/bin/bash
set -x

DATA_PATH="./datas/converted_hotpotqa_2000.json"
MODEL_NAME="Qwen3-1.7B"
MODEL_PATH="./models/${MODEL_NAME}"
OUTPUT_DIR="./output/NoMemoryGRPO1.7B"

ENV_TYPE="longcontext"
AGENT_TYPE="no_memory"

LR=5e-6
BETA=0.001
NUM_GENS=16
MAX_PROMPT_LEN=6000
MAX_GEN_LEN=2500
GRAD_ACC_STEPS=24
SAVE_STEPS=250

cd "$(dirname "$0")/.."

echo "Starting no-memory GRPO baseline..."
echo "Model: $MODEL_NAME"
echo "Env: $ENV_TYPE | Agent: $AGENT_TYPE"

python3 examples/train_mem_grpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --beta "$BETA" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --num_generations "$NUM_GENS" \
    --chunk_size 2500 \
    --max_chunk_number 6 \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_generate_length "$MAX_GEN_LEN" \
    --save_steps "$SAVE_STEPS" \
    --env_type "$ENV_TYPE" \
    --agent_type "$AGENT_TYPE" \
    --wandb_name "memfactory_${AGENT_TYPE}_${ENV_TYPE}_${MODEL_NAME}" \
    --epoch 2
