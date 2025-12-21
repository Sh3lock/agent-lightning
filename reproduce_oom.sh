#!/bin/bash
# 一键启动 Spider Pass@k 训练（修正版：并发启动版）

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

# ### 修改 1：更改短链接名称，防止覆盖第一个任务的链接 ###
SHORT_LINK="/tmp/r_${MY_USER}_job2" 
ln -snf "$REAL_RAY_DIR" "$SHORT_LINK"

export RAY_TMPDIR="$SHORT_LINK"

echo "=== 路径配置 ==="
echo "实际存放位置: $REAL_RAY_DIR"
echo "Ray 识别路径: $RAY_TMPDIR"

# --- 2. 核心参数设置 ---
TARGET_CONFIG="configs/reproduece_oom_2.json" 
TARGET_STAGE="--stage 1" 

# --- 自动化端口冲突处理 ---
get_free_port() {
    local port=$1
    while netstat -tln | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo "$port"
}

# ### 修改 2：彻底注释掉“清理残留进程”部分 ###
# (我们只想加一个任务，不想杀掉旧任务)
# my_pids=$(pgrep -u "$MY_USER" -f "python|ray|train_sql_agent" | grep -v $$)
# if [ -n "$my_pids" ]; then
#     echo "发送 SIGTERM 信号..."
#     echo "$my_pids" | xargs kill -15 2>/dev/null
#     sleep 5
# fi
# ... (后面的强杀逻辑也被略过)

# 4. 动态分配端口 (这个函数会自动寻找没被占用的端口，所以这里不需要改，它会自动+1)
SAFE_AGL_PORT=$(get_free_port 4750)
SAFE_RAY_PORT=$(get_free_port 8265)
export AGL_SERVER_PORT=$SAFE_AGL_PORT
export RAY_DASHBOARD_PORT=$SAFE_RAY_PORT

# --- 准备启动 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
cd "$SPIDER_DIR"

LOG_FILE="$SPIDER_DIR/log/train_reproduce_oom_2_${TS}.log"
mkdir -p "$SPIDER_DIR/log"

echo "Using Config: $TARGET_CONFIG"
echo "Using Log: $LOG_FILE"

# --- 正式启动 ---
export RAY_CHDIR_TO_TEMPDIR=1

# ### 修改 3：更换显卡 ID ###
# 原脚本是 6，如果你服务器有第 8 张卡(ID 7)，请改为 7。
# 如果你想让两任务挤在同一张卡上（极大概率会爆显存），请保持 6。
CUDA_VISIBLE_DEVICES=7 nohup python train_sql_agent.py local_qwen05 \
  --config-file "$TARGET_CONFIG" \
  $TARGET_STAGE \
  >> "$LOG_FILE" 2>&1 &

echo "----------------------------------------"
echo "第二个训练任务已启动！"
echo "显卡 ID: $CUDA_VISIBLE_DEVICES"
echo "Config: $TARGET_CONFIG"
echo "查看日志: tail -f $LOG_FILE"
echo "----------------------------------------"