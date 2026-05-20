#!/bin/bash
set -x

GPUS="${GPUS:-0}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-6000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
N_PATHS="${N_PATHS:-4}"
CONTEXT_TRUNCATION_SIDE="${CONTEXT_TRUNCATION_SIDE:-head}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

cd "$(dirname "$0")/.."

python3 evaluation/evaluate_orchestrator.py \
    --agent_type no_memory \
    --gpus "$GPUS" \
    --n_paths "$N_PATHS" \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_tokens "$MAX_GEN_LEN" \
    --context_truncation_side "$CONTEXT_TRUNCATION_SIDE" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
