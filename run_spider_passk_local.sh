#!/bin/bash
# 一键启动 Spider Pass@k 训练（Qwen2.5-Coder-0.5B，本地单机）
set -e
# 固定使用打好补丁的 venv Python
PYTHON_BIN="/home/lthpc/student/LiTengfei/env/light/bin/python"

# Python 搜索当前源码
export PYTHONPATH="$ROOT:$PYTHONPATH"
#############################################
# 可直接修改的硬编码参数
#############################################
STAGE=1                                # 1=Pass@k 探索；2=Pass@1 收敛
CUDA_VISIBLE_DEVICES="0"               # 选择要用的 GPU（如多卡可写成 "0,1"）
CONFIG_STAGE1="configs/passk_stage1_qwen05b.json"  # 相对 SPIDER_DIR
CONFIG_STAGE2="configs/passk_stage2_qwen05b.json"  # 相对 SPIDER_DIR
RESUME_CKPT=""                         # 如需从 stage1 恢复，填入 ckpt 路径；为空则不恢复
AGL_SERVER_PORT="${AGL_SERVER_PORT:-4750}"
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
cd "$SPIDER_DIR"

mkdir -p /home/lthpc/raytmp
export RAY_TMPDIR=/home/lthpc/raytmp
export VERL_SPIDER_DATA_DIR="$SPIDER_DIR/data"
export AGL_SERVER_PORT
# # 强制禁用 flash-attn，优先用 sdpa（或根据配置选 xformers）
# export TRANSFORMERS_ATTENTION_KERNEL=sdpa
# export PYTORCH_USE_FLASH_ATTENTION=0

if [[ "$STAGE" == "1" ]]; then
  CONFIG_PATH="$CONFIG_STAGE1"
  STAGE_ARG="--stage 1"
  CKPT_ARG=""
elif [[ "$STAGE" == "2" ]]; then
  CONFIG_PATH="$CONFIG_STAGE2"
  STAGE_ARG="--stage 2"
  CKPT_ARG=""
  if [[ -n "$RESUME_CKPT" ]]; then
    CKPT_ARG="--resume-ckpt $RESUME_CKPT"
  fi
else
  echo "Unknown STAGE=$STAGE (use 1 or 2)"
  exit 1
fi

CONFIG_STEM="$(basename "$CONFIG_PATH" .json)"
LOG_DIR="$SPIDER_DIR/log"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${CONFIG_STEM}_stage${STAGE}_${TS}.log"

{
  echo "========================================"
  echo "Training started at: $(date +"%Y-%m-%d %H:%M:%S")"
  echo "Hostname: $(hostname)"
  echo "User: $(whoami)"
  echo "CWD: $(pwd)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Python: $($PYTHON_BIN --version 2>&1)"
  echo "Config: $CONFIG_PATH"
  echo "Stage arg: $STAGE_ARG"
  echo "Resume ckpt: ${RESUME_CKPT:-none}"
  echo "========================================"
} > "$LOG_FILE"

# 可选：关闭 wandb
# export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" nohup "$PYTHON_BIN" train_sql_agent.py local_qwen05 \
  --config-file "$CONFIG_PATH" \
  $STAGE_ARG \
  $CKPT_ARG \
  >> "$LOG_FILE" 2>&1 &

echo "Training started with PID $!"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_FILE"
echo "Stage: $STAGE"
echo "Resume ckpt: ${RESUME_CKPT:-none}"
