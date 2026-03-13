#!/bin/bash
# 一键启动 Spider Pass@k 训练（点火版：Round-based guidance）

# --- 基础配置 ---
export WANDB_API_KEY="f093b349bddb4e99f211c7ba587579159de4e66b"
export WANDB_DIR=/home/storage/wenbinxing/ltf/passk/agent-lightning/wandb_logs
export VLLM_USE_V1=1
export RAY_DATA_DISK_USAGE_THRESHOLD=0.99

# 1. 路径管理：既要深目录的整洁，又要给 Ray 一个短路径
TS="$(date +%Y%m%d_%H%M%S)"
MY_USER=$(whoami)

# 这是你坚持要存放文件的真实长路径
REAL_RAY_DIR="/home/storage/wenbinxing/ltf/tmp/raytmp/ray_$TS"
mkdir -p "$REAL_RAY_DIR"

# 这是给 Ray 用的“短马甲”链接（放在 /tmp 下，名字极短）
SHORT_LINK="/tmp/r_${MY_USER}_round" 
ln -snf "$REAL_RAY_DIR" "$SHORT_LINK"

# 告诉 Ray 走“短马甲”路径
export RAY_TMPDIR="$SHORT_LINK"

echo "=== 路径配置 ==="
echo "实际存放位置: $REAL_RAY_DIR"
echo "Ray 识别路径: $RAY_TMPDIR"

# 2. 核心参数设置
STAGE=1
CONFIG_STAGE1="configs/passk_stage1_qwen05b_2epochs.json"
CONFIG_STAGE2="configs/passk_stage2_qwen05b.json"

# --- Round-based guidance 配置 ---
ROUND=0
OUTPUT_DIR="outputs/round0_full"
P_GUIDED=""  # 可选：覆盖默认退火，例如 1.0/0.5/0.25

# --- 自动化端口冲突处理 ---
get_free_port() {
    local port=$1
    while netstat -tln | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo "$port"
}

# 4. 动态分配端口
SAFE_AGL_PORT=$(get_free_port 4750)
SAFE_RAY_PORT=$(get_free_port 8265)
export AGL_SERVER_PORT=$SAFE_AGL_PORT
export RAY_DASHBOARD_PORT=$SAFE_RAY_PORT

# --- 准备启动 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
cd "$SPIDER_DIR"

if [[ "$STAGE" == "1" ]]; then
  CONFIG_PATH="$CONFIG_STAGE1"
  STAGE_ARG="--stage 1"
else
  CONFIG_PATH="$CONFIG_STAGE2"
  STAGE_ARG="--stage 2"
fi

# Round 产物路径
printf -v ROUND_PAD "%03d" "${ROUND}"
HARD_SAMPLES="${OUTPUT_DIR}/round_${ROUND}_hard_samples.jsonl"
GUIDANCE="${OUTPUT_DIR}/round_${ROUND}_guidance.jsonl"
IGNITE="${OUTPUT_DIR}/round_${ROUND}_ignite.jsonl"
RAW_SUCCESS_STATE="${OUTPUT_DIR}/raw_success_state.round${ROUND_PAD}.json"

# 简单检查
for f in "$HARD_SAMPLES" "$GUIDANCE" "$IGNITE" "$RAW_SUCCESS_STATE"; do
  if [[ ! -s "$f" ]]; then
    echo "缺少必要文件: $f"
    exit 1
  fi
done

# 日志文件名
LOG_FILE="$SPIDER_DIR/log/train_stage${STAGE}_round${ROUND}_${TS}.log"
mkdir -p "$SPIDER_DIR/log"

# --- 正式启动 ---
export RAY_CHDIR_TO_TEMPDIR=1

P_GUIDED_ARG=""
if [[ -n "$P_GUIDED" ]]; then
  P_GUIDED_ARG="--p-guided ${P_GUIDED}"
fi

# [注意] 这里目前指定了 GPU 0，如果需要更改请手动修改下面的数字
CUDA_VISIBLE_DEVICES=0 nohup python train_sql_agent.py local_qwen05 \
  --config-file "$CONFIG_PATH" \
  $STAGE_ARG \
  --round "$ROUND" \
  --hard-samples "$HARD_SAMPLES" \
  --guidance "$GUIDANCE" \
  --ignite "$IGNITE" \
  --raw-success-state "$RAW_SUCCESS_STATE" \
  $P_GUIDED_ARG \
  >> "$LOG_FILE" 2>&1 &

echo "----------------------------------------"
echo "训练已启动！"
echo "使用配置文件: $CONFIG_PATH"
echo "Round: $ROUND"
echo "查看日志: tail -f $LOG_FILE"
echo "----------------------------------------"
