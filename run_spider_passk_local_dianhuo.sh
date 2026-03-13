#!/bin/bash
# 一键启动 Spider Pass@k 训练（软链接版：解决路径过长 + 目录整洁）

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
# 每次运行都会更新这个链接指向最新的实验目录
SHORT_LINK="/tmp/r_${MY_USER}"
ln -snf "$REAL_RAY_DIR" "$SHORT_LINK"

# 告诉 Ray 走“短马甲”路径
export RAY_TMPDIR="$SHORT_LINK"

echo "=== 路径配置 ==="
echo "实际存放位置: $REAL_RAY_DIR"
echo "Ray 识别路径: $RAY_TMPDIR"

# 2. 核心参数设置
STAGE=1
# [修改点] 这里更新为了你要的 3steps 配置文件
CONFIG_STAGE1="configs/passk_stage1_qwen05b_3steps.json"
CONFIG_STAGE2="configs/passk_stage2_qwen05b.json"

# --- 自动化端口冲突处理 ---
get_free_port() {
    local port=$1
    while netstat -tln | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo "$port"
}

# 3. 清理残留进程 (保持你原本的注释状态)
# my_pids=$(pgrep -u "$MY_USER" -f "ray|train_sql_agent.py" | grep -v $$)
# if [ -n "$my_pids" ]; then
#     echo "发送 SIGTERM 信号..."
#     echo "$my_pids" | xargs kill -15 2>/dev/null
    
#     # 给驱动和显卡 10 秒钟的喘息时间来回收资源
#     echo "等待 10 秒让显存回收..."
#     echo "sleep 5"
#     sleep 5
# fi

# # 2. 如果还有残留，再使用强杀，但要先检查驱动是否还活着
# if timeout 2s nvidia-smi > /dev/null; then
#     remaining_pids=$(pgrep -u "$MY_USER" -f "python|ray|train_sql_agent" | grep -v $$)
#     if [ -n "$remaining_pids" ]; then
#         echo "清理顽固进程..."
#         echo "$remaining_pids" | xargs kill -9 2>/dev/null
#     fi
# else
#     echo "警告：检测到 GPU 驱动响应异常，停止启动脚本，请检查硬件！"
#     exit 1
# fi

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

# 日志文件名增加了 Config 标识，方便区分
LOG_FILE="$SPIDER_DIR/log/train_stage${STAGE}_3steps_${TS}.log"
mkdir -p "$SPIDER_DIR/log"

# --- 正式启动 ---
export RAY_CHDIR_TO_TEMPDIR=1

# [注意] 这里目前指定了 GPU 6，如果需要更改请手动修改下面的数字
CUDA_VISIBLE_DEVICES=6 nohup python train_sql_agent.py local_qwen05 \
  --config-file "$CONFIG_PATH" \
  $STAGE_ARG \
  >> "$LOG_FILE" 2>&1 &

echo "----------------------------------------"
echo "训练已启动！"
echo "使用配置文件: $CONFIG_PATH"
echo "你可以去这里查看详细的 Ray 运行文件（已分类）:"
echo "cd $REAL_RAY_DIR"
echo "查看日志: tail -f $LOG_FILE"
echo "----------------------------------------"