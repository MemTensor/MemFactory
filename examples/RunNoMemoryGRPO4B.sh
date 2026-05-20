#!/bin/bash
set -x

DATA_PATH="./datas/converted_hotpotqa_2000.json"
MODEL_NAME="Qwen3-4B-Instruct"
MODEL_ROOT="${MODEL_ROOT:-./models}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
if [ ! -d "$MODEL_PATH" ] && [ -d "/mnt/afs/models/${MODEL_NAME}" ]; then
    MODEL_PATH="/mnt/afs/models/${MODEL_NAME}"
fi
OUTPUT_DIR="./output/NoMemoryGRPO4B"

ENV_TYPE="longcontext"
AGENT_TYPE="no_memory"

LR="${LR:-5e-6}"
BETA="${BETA:-0.001}"
NUM_GENS="${NUM_GENS:-16}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-5500}"
MAX_GEN_LEN="${MAX_GEN_LEN:-1500}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-24}"
SAVE_STEPS="${SAVE_STEPS:-250}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-1}"
CONTEXT_TRUNCATION_SIDE="${CONTEXT_TRUNCATION_SIDE:-head}"

cd "$(dirname "$0")/.."

echo "Starting no-memory GRPO baseline..."
echo "Model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Env: $ENV_TYPE | Agent: $AGENT_TYPE"

python3 examples/train_mem_grpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --beta "$BETA" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --train_micro_batch_size "$TRAIN_MICRO_BATCH_SIZE" \
    --logprob_batch_size "$LOGPROB_BATCH_SIZE" \
    --num_generations "$NUM_GENS" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --chunk_size 2500 \
    --max_chunk_number 6 \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_generate_length "$MAX_GEN_LEN" \
    --context_truncation_side "$CONTEXT_TRUNCATION_SIDE" \
    --save_steps "$SAVE_STEPS" \
    --env_type "$ENV_TYPE" \
    --agent_type "$AGENT_TYPE" \
    --wandb_name "memfactory_${AGENT_TYPE}_${ENV_TYPE}_${MODEL_NAME}" \
    --epoch 2
