#!/bin/bash
# 一键启动 Spider Pass@k 训练（Qwen2.5-Coder-0.5B，本地单机）
# 固定使用打好补丁的 venv Python

#############################################
# 可直接修改的硬编码参数
#############################################
export PYTHONPATH=/root/verl_pass/verl:$PYTHONPATH
STAGE=1                                # 1=Pass@k 探索；2=Pass@1 收敛
CUDA_VISIBLE_DEVICES="0"               # 选择要用的 GPU（如多卡可写成 "0,1"）
CONFIG_STAGE1="configs/passk_stage1_qwen05b.json"  # 相对 SPIDER_DIR
CONFIG_STAGE2="configs/passk_stage2_qwen05b.json"  # 相对 SPIDER_DIR
RESUME_CKPT="/root/autodl-tmp/ckpt_passk_stage1/global_step_15"                         # 如需从 stage1 恢复，填入 ckpt 路径；为空则不恢复
AGL_SERVER_PORT="${AGL_SERVER_PORT:-4750}"
#############################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
cd "$SPIDER_DIR"

export VERL_SPIDER_DATA_DIR="$SPIDER_DIR/data"
# 启动前检查和清理
echo "=== 启动前检查和清理 ==="
echo "检查是否有残留的进程和端口占用..."

# 检查 Python 进程
python_processes=$(ps aux | grep -E "(python|train_sql_agent)" | grep -v grep | grep -v run_spider_passk_local_test.sh)
if [ ! -z "$python_processes" ]; then
    echo "发现残留的 Python 进程："
    echo "$python_processes"
    echo "正在清理..."
    
    # 提取PID并终止进程
    echo "$python_processes" | awk '{print $2}' | xargs -r kill -TERM
    
    # 等待进程优雅退出
    sleep 3
    
    # 检查是否还有残留进程，强制终止
    remaining_processes=$(ps aux | grep -E "(python|train_sql_agent)" | grep -v grep | grep -v run_spider_passk_local_test.sh)
    if [ ! -z "$remaining_processes" ]; then
        echo "强制终止残留进程..."
        echo "$remaining_processes" | awk '{print $2}' | xargs -r kill -KILL
    fi
else
    echo "未发现残留的 Python 进程"
fi

# 检查常见端口占用
ports=(4789 8000 8001 8002 8080 8888)
for port in "${ports[@]}"; do
    port_process=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}')
    if [ ! -z "$port_process" ] && [ "$port_process" != "-" ]; then
        echo "发现端口 $port 被占用: $port_process"
        pid=$(echo "$port_process" | cut -d'/' -f1)
        if [ ! -z "$pid" ]; then
            echo "正在终止占用端口 $port 的进程 (PID: $pid)..."
            kill -TERM "$pid" 2>/dev/null
            sleep 1
            kill -KILL "$pid" 2>/dev/null
        fi
    fi
done

echo "清理完成，等待2秒后启动..."
sleep 1

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

nohup python train_sql_agent.py local_qwen05 \
  --config-file "$CONFIG_PATH" \
  $STAGE_ARG \
  $CKPT_ARG \
  >> "$LOG_FILE" 2>&1 &

echo "Training started with PID $!"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_FILE"
echo "Stage: $STAGE"
echo "Resume ckpt: ${RESUME_CKPT:-none}"
