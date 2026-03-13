#!/bin/bash
# Train GRPO / Pass@k / Pass@k+Guidance with aligned config2 settings.
#
# Usage examples:
#   MODE=smoke RUN_PARALLEL=0 bash run_spider_config2_train_matrix.sh
#   MODE=full RUN_SET=passk,guided RUN_PARALLEL=1 bash run_spider_config2_train_matrix.sh

set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/storage/wenbinxing/ltf}"
RAY_TMPDIR_FALLBACK_BASE="${RAY_TMPDIR_FALLBACK_BASE:-/home/storage/wenbinxing/ltf/tmp}"
if [[ -z "${RAY_TMPDIR_BASE:-}" ]]; then
  default_ray_tmp="$ARTIFACT_ROOT/tmp"
  fs_use="$(df -P "$ARTIFACT_ROOT" 2>/dev/null | awk 'NR==2 {gsub("%", "", $5); print $5}')"
  if [[ -n "$fs_use" && "$fs_use" -ge 95 ]]; then
    RAY_TMPDIR_BASE="$RAY_TMPDIR_FALLBACK_BASE"
  else
    RAY_TMPDIR_BASE="$default_ray_tmp"
  fi
fi
MODE="${MODE:-smoke}"                 # smoke | full
RUN_SET="${RUN_SET:-grpo,passk,guided}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"     # 0 sequential, 1 background
PYTHON_BIN="${PYTHON_BIN:-/home/wenbinxing/anaconda3/envs/ltf_agent/bin/python}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
SPIDER_USE_REMOVE_PADDING="${SPIDER_USE_REMOVE_PADDING:-0}"
SPIDER_USE_TORCH_COMPILE="${SPIDER_USE_TORCH_COMPILE:-1}"
SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-}"
SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE="${SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE:-4}"
SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION="${SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION:-}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
RAY_LOCAL_FS_CAPACITY_THRESHOLD="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"

BASE_STORE_PORT="${BASE_STORE_PORT:-4747}"
BASE_AGENT_PORT="${BASE_AGENT_PORT:-9999}"

GPU_GRPO="${GPU_GRPO:-0}"
GPU_PASSK="${GPU_PASSK:-0}"
GPU_GUIDED="${GPU_GUIDED:-0}"

ROUND="${ROUND:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/round0_full}"
P_GUIDED="${P_GUIDED:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_DIR="$SCRIPT_DIR/examples/spider"
CONFIG_REL="configs/passk_stage1_qwen05b_config2_2epochs.json"
CONFIG_ABS="$SPIDER_DIR/$CONFIG_REL"

if [[ ! -f "$CONFIG_ABS" ]]; then
  echo "missing config: $CONFIG_ABS"
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python binary not found or not executable: $PYTHON_BIN"
  exit 1
fi

COMMON_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  COMMON_ARGS+=(--total-epochs 1 --total-training-steps 30)
elif [[ "$MODE" != "full" ]]; then
  echo "unsupported MODE: $MODE (expected smoke/full)"
  exit 1
fi

contains() {
  local token="$1"
  [[ ",$RUN_SET," == *",$token,"* ]]
}

start_run() {
  local run_name="$1"
  local gpu="$2"
  local port_offset="$3"
  shift 3
  local -a extra_args=("$@")

  local run_root="$ARTIFACT_ROOT/passk/agent-lightning/examples/spider/log/$run_name"
  local ckpt_root="$ARTIFACT_ROOT/passk/agent-lightning/examples/spider/ckpt/$run_name"
  local ray_root="$ARTIFACT_ROOT/passk/agent-lightning/examples/spider/ray/$run_name"
  mkdir -p "$run_root" "$ckpt_root" "$ray_root" "$RAY_TMPDIR_BASE"
  local store_port=$((BASE_STORE_PORT + port_offset))
  local agent_port=$((BASE_AGENT_PORT + port_offset))

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local launcher_log="$run_root/launcher_${ts}.log"
  local ray_tmp="$RAY_TMPDIR_BASE"

  echo "----------------------------------------"
  echo "run_name: $run_name"
  echo "gpu: $gpu"
  echo "mode: $MODE"
  echo "run_root: $run_root"
  echo "ckpt_root: $ckpt_root"
  echo "python_bin: $PYTHON_BIN"
  echo "AGL_SERVER_PORT: $store_port"
  echo "agentlightning_port: $agent_port"
  echo "ray_tmpdir: $ray_tmp"
  echo "launcher_log: $launcher_log"
  echo "----------------------------------------"

  local -a cmd=(
    "$PYTHON_BIN" train_sql_agent.py local_qwen05
    --config-file "$CONFIG_ABS"
    --stage 1
    --agentlightning-port "$agent_port"
    "${COMMON_ARGS[@]}"
    "${extra_args[@]}"
  )

  {
    echo "========================================"
    echo "started_at: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "run_name: $run_name"
    echo "mode: $MODE"
    echo "gpu: $gpu"
    echo "cwd: $SPIDER_DIR"
    echo "python_bin: $PYTHON_BIN"
    echo "AGL_SERVER_PORT: $store_port"
    echo "agentlightning_port: $agent_port"
    echo "SPIDER_RUN_ROOT: $run_root"
    echo "SPIDER_CKPT_ROOT: $ckpt_root"
    echo "SPIDER_RAY_ROOT: $ray_root"
    echo "RAY_TMPDIR: $ray_tmp"
    echo "RAY_TMPDIR_BASE: $RAY_TMPDIR_BASE"
    echo "RAY_TMPDIR_FALLBACK_BASE: $RAY_TMPDIR_FALLBACK_BASE"
    echo "VLLM_USE_V1: $VLLM_USE_V1"
    echo "SPIDER_USE_REMOVE_PADDING: $SPIDER_USE_REMOVE_PADDING"
    echo "SPIDER_USE_TORCH_COMPILE: $SPIDER_USE_TORCH_COMPILE"
    echo "SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU: ${SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-<unset>}"
    echo "SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE: $SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE"
    echo "SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION: ${SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION:-<unset>}"
    echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
    echo "RAY_DISABLE_DASHBOARD: $RAY_DISABLE_DASHBOARD"
    echo "RAY_LOCAL_FS_CAPACITY_THRESHOLD: $RAY_LOCAL_FS_CAPACITY_THRESHOLD"
    echo "git_branch: $(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "git_commit: $(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "cmd: ${cmd[*]}"
    echo "========================================"
  } > "$launcher_log"

  if [[ "$RUN_PARALLEL" == "1" ]]; then
    (
      cd "$SPIDER_DIR"
      setsid nohup env \
        SPIDER_RUN_ROOT="$run_root" \
        SPIDER_CKPT_ROOT="$ckpt_root" \
        SPIDER_RAY_ROOT="$ray_root" \
        SPIDER_ISOLATE_CKPT=1 \
        SPIDER_LOG_ROLLOUT_INFO="${SPIDER_LOG_ROLLOUT_INFO:-0}" \
        SPIDER_SAVE_RAW_LOG="${SPIDER_SAVE_RAW_LOG:-1}" \
        SPIDER_RAW_LOG_LEVEL="${SPIDER_RAW_LOG_LEVEL:-INFO}" \
        SPIDER_USE_REMOVE_PADDING="$SPIDER_USE_REMOVE_PADDING" \
        SPIDER_USE_TORCH_COMPILE="$SPIDER_USE_TORCH_COMPILE" \
        SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="$SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
        SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE="$SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE" \
        SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION="$SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION" \
        VLLM_USE_V1="$VLLM_USE_V1" \
        CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
        RAY_DISABLE_DASHBOARD="$RAY_DISABLE_DASHBOARD" \
        RAY_LOCAL_FS_CAPACITY_THRESHOLD="$RAY_LOCAL_FS_CAPACITY_THRESHOLD" \
        RAY_local_fs_capacity_threshold="$RAY_LOCAL_FS_CAPACITY_THRESHOLD" \
        RAY_DATA_DISK_USAGE_THRESHOLD=0.99 \
        RAY_CHDIR_TO_TEMPDIR=1 \
        RAY_TMPDIR="$ray_tmp" \
        TMPDIR="$ray_tmp" \
        PYTHONUNBUFFERED=1 \
        AGL_SERVER_PORT="$store_port" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        "${cmd[@]}" >> "$launcher_log" 2>&1 < /dev/null &
      echo "$!" > "$run_root/latest.pid"
    )
    local pid
    pid="$(cat "$run_root/latest.pid")"
    echo "started in background, pid=$pid"
  else
    (
      cd "$SPIDER_DIR"
      env \
        SPIDER_RUN_ROOT="$run_root" \
        SPIDER_CKPT_ROOT="$ckpt_root" \
        SPIDER_RAY_ROOT="$ray_root" \
        SPIDER_ISOLATE_CKPT=1 \
        SPIDER_LOG_ROLLOUT_INFO="${SPIDER_LOG_ROLLOUT_INFO:-0}" \
        SPIDER_SAVE_RAW_LOG="${SPIDER_SAVE_RAW_LOG:-1}" \
        SPIDER_RAW_LOG_LEVEL="${SPIDER_RAW_LOG_LEVEL:-INFO}" \
        SPIDER_USE_REMOVE_PADDING="$SPIDER_USE_REMOVE_PADDING" \
        SPIDER_USE_TORCH_COMPILE="$SPIDER_USE_TORCH_COMPILE" \
        SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="$SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
        SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE="$SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE" \
        SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION="$SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION" \
        VLLM_USE_V1="$VLLM_USE_V1" \
        CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING" \
        RAY_DISABLE_DASHBOARD="$RAY_DISABLE_DASHBOARD" \
        RAY_LOCAL_FS_CAPACITY_THRESHOLD="$RAY_LOCAL_FS_CAPACITY_THRESHOLD" \
        RAY_local_fs_capacity_threshold="$RAY_LOCAL_FS_CAPACITY_THRESHOLD" \
        RAY_DATA_DISK_USAGE_THRESHOLD=0.99 \
        RAY_CHDIR_TO_TEMPDIR=1 \
        RAY_TMPDIR="$ray_tmp" \
        TMPDIR="$ray_tmp" \
        PYTHONUNBUFFERED=1 \
        AGL_SERVER_PORT="$store_port" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        "${cmd[@]}" 2>&1 | tee -a "$launcher_log"
    )
  fi
}

if contains "grpo"; then
  start_run "config2_grpo" "$GPU_GRPO" 0 --adv-estimator grpo
fi

if contains "passk"; then
  start_run "config2_passk" "$GPU_PASSK" 1 --adv-estimator grpo_passk_seed
fi

if contains "guided"; then
  HARD_SAMPLES="$SPIDER_DIR/$OUTPUT_DIR/round_${ROUND}_hard_samples.jsonl"
  GUIDANCE="$SPIDER_DIR/$OUTPUT_DIR/round_${ROUND}_guidance.jsonl"
  IGNITE="$SPIDER_DIR/$OUTPUT_DIR/round_${ROUND}_ignite.jsonl"
  printf -v ROUND_PAD "%03d" "$ROUND"
  RAW_SUCCESS_STATE="$SPIDER_DIR/$OUTPUT_DIR/raw_success_state.round${ROUND_PAD}.json"

  for f in "$HARD_SAMPLES" "$GUIDANCE" "$IGNITE" "$RAW_SUCCESS_STATE"; do
    if [[ ! -s "$f" ]]; then
      echo "missing guidance artifact: $f"
      exit 1
    fi
  done

  GUIDED_ARGS=(
    --adv-estimator grpo_passk_seed
    --round "$ROUND"
    --hard-samples "$HARD_SAMPLES"
    --guidance "$GUIDANCE"
    --ignite "$IGNITE"
    --raw-success-state "$RAW_SUCCESS_STATE"
  )
  if [[ -n "$P_GUIDED" ]]; then
    GUIDED_ARGS+=(--p-guided "$P_GUIDED")
  fi

  start_run "config2_passk_guided_round${ROUND}" "$GPU_GUIDED" 2 "${GUIDED_ARGS[@]}"
fi

if [[ "$RUN_PARALLEL" == "1" ]]; then
  echo "all requested runs are in background"
  echo "check logs under: $ARTIFACT_ROOT/passk/agent-lightning/examples/spider/log"
fi
