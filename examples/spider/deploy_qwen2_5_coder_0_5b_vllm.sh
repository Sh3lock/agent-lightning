#!/bin/bash

# Qwen2.5-Coder-0.5B Single GPU Deployment Script
# TARGET HARDWARE: 48GB VRAM GPU (e.g., A6000, A40, L40)
# STRATEGY: Maximum Concurrency & Throughput

set -euo pipefail

# Activate conda environment when available
if command -v conda &> /dev/null; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
else
    echo "Error: conda not found and no conda.sh in \$HOME/anaconda3 or \$HOME/miniconda3."
    exit 1
fi

if [[ -n "${VLLM_CONDA_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$VLLM_CONDA_ENV"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
VLLM_PATH="${VLLM_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${SPIDER_MODEL_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}}"

# --- GPU Configuration ---
: "${CUDA_VISIBLE_DEVICES:=0}"  # Override before running if needed

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
DEPLOYMENT_TYPE="Single-GPU-48GB-HighLoad"

# --- EXTREME Performance Parameters for 48GB VRAM ---

# 1. Concurrency: Pushed to 256
#    48GB VRAM allows huge KV cache. High concurrency is the ONLY way
#    to keep GPU utilization high during the Decode phase for a 0.5B model.
#    (If client sends enough requests, you can even try 512)
MAX_NUM_SEQS=256

# 2. Batching: Maxed out to 65536
#    Processing massive amounts of prompt tokens in one go.
MAX_NUM_BATCHED_TOKENS=65536

# 3. Memory: Set to 0.90
#    10% of 48GB is ~4.8GB. This is a massive safety buffer for a 0.5B model.
#    OOM is extremely unlikely with this setting.
GPU_MEMORY_UTILIZATION=0.90

# 4. Context Length
#    If your inputs are long, ensure this covers them.
MAX_MODEL_LEN=8192

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder-0.5b-instruct}"

echo "========================================"
echo "Qwen2.5-Coder-0.5B 48GB EXTREME DEPLOYMENT"
echo "========================================"
echo "Using GPUs: ${GPUS[*]}"
echo "Strategy: Massive Concurrency"
echo "Max Concurrent Sequences: $MAX_NUM_SEQS (Doubled)"
echo "Max Batched Tokens: $MAX_NUM_BATCHED_TOKENS (Increased)"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "========================================"

# Check if required directories exist
if [[ -n "$VLLM_PATH" ]] && [ ! -d "$VLLM_PATH" ]; then
    echo "Error: vLLM directory not found at $VLLM_PATH"
    exit 1
fi

if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    exit 1
fi

# Verify GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found."
    exit 1
fi

echo ""
echo "Starting vLLM server..."
echo "Running in high-throughput mode..."
echo ""

if [[ -n "$VLLM_PATH" ]]; then
    export PYTHONPATH="$VLLM_PATH:${PYTHONPATH:-}"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --dtype auto \
    --disable-log-requests
    # --enforce-eager  <-- REMOVED. Always use CUDA Graphs.

echo "Deployment completed!"
