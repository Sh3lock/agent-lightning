#!/bin/bash
# One-shot pipeline: mine hard samples -> generate guidance -> ignite guidance
# Configure variables below as needed.

set -euo pipefail

# ===== User Config (override with envs if needed) =====
ROUND=0
INPUT_DATA="${INPUT_DATA:-data/train_spider.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/round0_full}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${SPIDER_MODEL_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}}"
TMP_BASE="${TMP_BASE:-$(pwd)/tmp}"

# Mining settings
HARD_K="${HARD_K:-6}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"  # -1 means all samples in the input dataset
MINER_MODEL="${MINER_MODEL:-qwen2.5-coder-0.5b-instruct}"
MINER_WORKERS="${MINER_WORKERS:-64}"

# Strong model settings (guidance generation)
STRONG_API_BASE="${STRONG_API_BASE:-}"
STRONG_API_KEY="${STRONG_API_KEY:-}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-235b-a22b-instruct-2507}"
GUIDANCE_WORKERS="${GUIDANCE_WORKERS:-64}"
GUIDANCE_REQUEST_INTERVAL="${GUIDANCE_REQUEST_INTERVAL:-0.2}"
RESUME_IF_POSSIBLE="${RESUME_IF_POSSIBLE:-1}"

# Ignite (vLLM) settings
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8001/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
IGNITE_K="${IGNITE_K:-4}"
IGNITE_MODEL="${IGNITE_MODEL:-qwen2.5-coder-0.5b-instruct}"
IGNITE_WORKERS="${IGNITE_WORKERS:-32}"

# ===== Derived paths =====
HARD_SAMPLES="${OUTPUT_DIR}/round_${ROUND}_hard_samples.jsonl"
GUIDANCE_OUT="${OUTPUT_DIR}/round_${ROUND}_guidance.jsonl"
IGNITE_OUT="${OUTPUT_DIR}/round_${ROUND}_ignite.jsonl"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TMP_BASE}"
export TMPDIR="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TMP="${TMP_BASE}"

MINE_RESUME_FLAGS=()
GUIDANCE_RESUME_FLAGS=()
IGNITE_RESUME_FLAGS=()
if [[ "${RESUME_IF_POSSIBLE}" -eq 1 ]]; then
  MINE_RESUME_FLAGS+=(--resume)
  GUIDANCE_RESUME_FLAGS+=(--resume)
  IGNITE_RESUME_FLAGS+=(--resume)
fi

if [[ -z "${STRONG_API_BASE}" || -z "${STRONG_API_KEY}" ]]; then
  echo "Set STRONG_API_BASE and STRONG_API_KEY before generating guidance." >&2
  exit 1
fi

echo "[1/3] Mining hard samples..."
export OPENAI_API_BASE="${VLLM_API_BASE}"
export OPENAI_API_KEY="${VLLM_API_KEY}"
python scripts/mine_hard_round.py \
  --input "${INPUT_DATA}" \
  --round "${ROUND}" \
  --output-dir "${OUTPUT_DIR}" \
  --k "${HARD_K}" \
  --num-samples "${NUM_SAMPLES}" \
  --model "${MINER_MODEL}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --endpoint "${OPENAI_API_BASE}" \
  --dump-gold-for-debug \
  --num-workers "${MINER_WORKERS}" \
  "${MINE_RESUME_FLAGS[@]}"

echo "[2/3] Generating guidance (L1/L2) with strong model..."
python scripts/generate_guidance.py \
  --hard-samples "${HARD_SAMPLES}" \
  --output "${GUIDANCE_OUT}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --guidance-model "${STRONG_MODEL}" \
  --guidance-endpoint "${STRONG_API_BASE}" \
  --guidance-api-key "${STRONG_API_KEY}" \
  --num-workers "${GUIDANCE_WORKERS}" \
  --request-interval "${GUIDANCE_REQUEST_INTERVAL}" \
  "${GUIDANCE_RESUME_FLAGS[@]}"

echo "[3/3] Ignite guidance on vLLM (L1->L2)..."
python scripts/ignite_guidance.py \
  --hard-samples "${HARD_SAMPLES}" \
  --guidance "${GUIDANCE_OUT}" \
  --output "${IGNITE_OUT}" \
  --round "${ROUND}" \
  --k "${IGNITE_K}" \
  --model "${IGNITE_MODEL}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --endpoint "${OPENAI_API_BASE}" \
  --dataset "${INPUT_DATA}" \
  --num-workers "${IGNITE_WORKERS}" \
  "${IGNITE_RESUME_FLAGS[@]}"

echo "Done. Outputs:"
echo "  - ${HARD_SAMPLES}"
echo "  - ${GUIDANCE_OUT}"
echo "  - ${IGNITE_OUT}"
