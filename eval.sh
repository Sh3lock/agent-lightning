#!/bin/bash

# Run spider SQL agent against the locally deployed GRPO fine-tuned model API.
# Only the API endpoint/model/key are overridden to point to local service.
ROOT=/home/storage/wenbinxing/ltf/passk/agent-lightning
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Spider 数据目录（设置为绝对路径，避免相对路径找不到数据）
export VERL_SPIDER_DATA_DIR="$SCRIPT_DIR/examples/spider/data"

# 运行评估（可按需调整参数）
mkdir -p outputs
python examples/spider/sql_agent_eval.py \
  --mode eval \
  --num-samples -1 \
  --output outputs/grpo-finetuned-model-dev-local.jsonl \
  --concurrency 60 \
  --pass-k 4