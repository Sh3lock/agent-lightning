#!/bin/bash
# 一键启动 Spider Pass@k 训练（多任务并行版 - Config2）

# --- 基础配置 ---
export WANDB_API_KEY="f093b349bddb4e99f211c7ba587579159de4e66b"
export WANDB_DIR=/home/storage/wenbinxing/ltf/passk/agent-lightning/wandb_logs
export VLLM_USE_V1=1
export RAY_DATA_DISK_USAGE_THRESHOLD=0.98

# 1. 路径管理
TS="$(date +%Y%m%d_%H%M%S)"
MY_USER=$(whoami)

# 真实长路径
REAL_RAY_DIR="/home/storage/wenbinxing/ltf/tmp/raytmp/ray_$TS"
mkdir -p "$REAL_RAY_DIR"

# ### 修改 1：更改 Ray 的短链接名称 ###
# 必须改名（例如加上 _cfg2），否则会覆盖掉你正在运行的第一个任务的链接，导致 Ray 报错
SHORT_LINK="/tmp/r_${MY_USER}_cfg2"
ln -snf "$REAL_RAY_DIR" "$SHORT_LINK"

# 告诉 Ray 走新的“短马甲”路径
export RAY_TMPDIR="$SHORT_LINK"

echo "=== 路径配置 ==="
echo "实际存放位置: $REAL_RAY_DIR"
echo "Ray 识别路径: $RAY_TMPDIR"

# ### 修改 2：指定新的配置文件路径 ###
# 直接填入你提供的绝对路径，确保无误
CONFIG_PATH="/home/storage/wenbinxing/ltf/passk/agent-lightning/examples/spider/configs/passk_stage1_qwen05b_config2.json"
# 根据文件名判断这是 stage 1
STAGE_ARG="--stage 1"

# --- 自动化端口冲突处理 ---
get_free_port() {
    local port=$1
    while netstat -tln | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo "$port"
}

# 3. 清理残留进程 (保持注释状态，绝对不要打开，否则会杀掉第一个任务)
# my_pids=$(pgrep -u "$MY_USER" -f "ray|train_sql_agent.py" | grep -v $$)
# ... (代码略) ...

# 4. 动态分配端口 (自动寻找空闲端口，无需修改)
SAFE_AGL_PORT=$(get_free_port 4750)
SAFE_RAY_PORT=$(get_free_port 8265)
export AGL_SERVER_PORT=$SAFE_AGL_PORT
export RAY_DASHBOARD_PORT=$SAFE_RAY_PORT

# --- 准备启动 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
cd "$SPIDER_DIR"

# 日志文件名加上 config2 标识，方便区分
LOG_FILE="$SPIDER_DIR/log/train_stage1_config2_${TS}.log"
mkdir -p "$SPIDER_DIR/log"

echo "Using Config: $CONFIG_PATH"
echo "Log File: $LOG_FILE"

# --- 正式启动 ---
export RAY_CHDIR_TO_TEMPDIR=1

# ### 修改 3：更换显卡 ID ###
# 请先运行 nvidia-smi 确认哪张卡是空的。
# 假设之前的任务跑在 6 号卡，这里我改为 7 号卡。
# 如果你只有一张卡，且显存不够跑两个任务，这里会报错。
CUDA_VISIBLE_DEVICES=7 nohup python train_sql_agent.py local_qwen05 \
  --config-file "$CONFIG_PATH" \
  $STAGE_ARG \
  >> "$LOG_FILE" 2>&1 &

echo "----------------------------------------"
echo "新训练任务 (Config 2) 已启动！"
echo "显卡 ID: $CUDA_VISIBLE_DEVICES"
echo "Ray 路径: $RAY_TMPDIR"
echo "查看日志: tail -f $LOG_FILE"
echo "----------------------------------------"