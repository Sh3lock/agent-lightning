#!/bin/bash
# Evaluate multiple checkpoints on dev500 + hard_raw first, then optional full test set.
#
# Required envs (set at least one):
#   GRPO_CKPT=/path/to/grpo/best
#   PASSK_CKPT=/path/to/passk/best
#   GUIDED_CKPT=/path/to/guided/best
#
# Optional:
#   RUN_FULL_TEST=1  # also evaluate on data/test.parquet
#   GPU_EVAL=0
#   ARTIFACT_ROOT=/abs/path/to/output_root
#   EVAL_AGENTLIGHTNING_PORT_BASE=12000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/.artifacts}"
RAY_TMPDIR_FALLBACK_BASE="${RAY_TMPDIR_FALLBACK_BASE:-$ARTIFACT_ROOT/tmp_fallback}"
if [[ -z "${RAY_TMPDIR_BASE:-}" ]]; then
  default_ray_tmp="$ARTIFACT_ROOT/passk/tmp"
  fs_use="$(df -P "$ARTIFACT_ROOT" 2>/dev/null | awk 'NR==2 {gsub("%", "", $5); print $5}')"
  if [[ -n "$fs_use" && "$fs_use" -ge 95 ]]; then
    RAY_TMPDIR_BASE="$RAY_TMPDIR_FALLBACK_BASE"
  else
    RAY_TMPDIR_BASE="$default_ray_tmp"
  fi
fi
RUN_FULL_TEST="${RUN_FULL_TEST:-0}"
GPU_EVAL="${GPU_EVAL:-0}"
EVAL_AGENTLIGHTNING_PORT_BASE="${EVAL_AGENTLIGHTNING_PORT_BASE:-12000}"
EVAL_PORT_COUNTER=0
VLLM_USE_V1="${VLLM_USE_V1:-1}"
SPIDER_USE_REMOVE_PADDING="${SPIDER_USE_REMOVE_PADDING:-0}"
SPIDER_USE_TORCH_COMPILE="${SPIDER_USE_TORCH_COMPILE:-1}"
SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-}"
SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE="${SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE:-4}"
SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION="${SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION:-}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
RAY_LOCAL_FS_CAPACITY_THRESHOLD="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"
LOCAL_NO_PROXY_DEFAULT="localhost,127.0.0.1,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  PROXY_NO_PROXY="${NO_PROXY},${LOCAL_NO_PROXY_DEFAULT}"
else
  PROXY_NO_PROXY="$LOCAL_NO_PROXY_DEFAULT"
fi

SPIDER_DIR="$SCRIPT_DIR/examples/spider"
RESULT_ROOT="$ARTIFACT_ROOT/examples/spider/eval_matrix/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_ROOT"

CFG_DEV500="configs/passk_stage1_qwen05b_config2_eval_test_dev500.json"
CFG_HARD="configs/passk_stage1_qwen05b_config2_eval_hard_raw.json"
CFG_TEST="configs/passk_stage1_qwen05b_config2_eval_test.json"

SUMMARY="$RESULT_ROOT/summary.tsv"
echo -e "label\tsplit\tsuccess\treward\tstatus\trun_dir" > "$SUMMARY"

run_eval() {
  local label="$1"
  local ckpt="$2"
  local split="$3"
  local cfg="$4"

  if [[ -z "$ckpt" ]]; then
    return
  fi
  if [[ ! -d "$ckpt" ]]; then
    echo "[$label/$split] ckpt missing: $ckpt"
    echo -e "${label}\t${split}\tNA\tNA\tmissing_ckpt\t-" >> "$SUMMARY"
    return
  fi

  local run_root="$RESULT_ROOT/log_${label}_${split}"
  local ckpt_root="$RESULT_ROOT/ckpt_tmp_${label}_${split}"
  local ray_root="$RESULT_ROOT/ray_${label}_${split}"
  local eval_port=$((EVAL_AGENTLIGHTNING_PORT_BASE + EVAL_PORT_COUNTER))
  EVAL_PORT_COUNTER=$((EVAL_PORT_COUNTER + 1))
  local ray_tmp="$RAY_TMPDIR_BASE"
  mkdir -p "$run_root" "$ckpt_root" "$ray_root" "$ray_tmp"

  local launcher_log="$run_root/launcher.log"
  echo "[$label/$split] start eval with ckpt=$ckpt"
  {
    echo "========================================"
    echo "started_at: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "label: $label"
    echo "split: $split"
    echo "ckpt: $ckpt"
    echo "config: $cfg"
    echo "gpu: $GPU_EVAL"
    echo "agentlightning_port: $eval_port"
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
    echo "========================================"
  } > "$launcher_log"

  (
    export SPIDER_RUN_ROOT="$run_root"
    export SPIDER_CKPT_ROOT="$ckpt_root"
    export SPIDER_RAY_ROOT="$ray_root"
    export SPIDER_ISOLATE_CKPT=1
    export SPIDER_LOG_ROLLOUT_INFO=0
    export SPIDER_SAVE_RAW_LOG="${SPIDER_SAVE_RAW_LOG:-1}"
    export SPIDER_RAW_LOG_LEVEL="${SPIDER_RAW_LOG_LEVEL:-INFO}"
    export SPIDER_USE_REMOVE_PADDING="$SPIDER_USE_REMOVE_PADDING"
    export SPIDER_USE_TORCH_COMPILE="$SPIDER_USE_TORCH_COMPILE"
    export SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="$SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
    export SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE="$SPIDER_SAFE_LOGPROB_MB_WHEN_DENSE"
    export SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION="$SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION"
    export VLLM_USE_V1="$VLLM_USE_V1"
    export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"
    export RAY_DISABLE_DASHBOARD="$RAY_DISABLE_DASHBOARD"
    export RAY_LOCAL_FS_CAPACITY_THRESHOLD="$RAY_LOCAL_FS_CAPACITY_THRESHOLD"
    export RAY_local_fs_capacity_threshold="$RAY_LOCAL_FS_CAPACITY_THRESHOLD"
    export RAY_DATA_DISK_USAGE_THRESHOLD=0.99
    export RAY_CHDIR_TO_TEMPDIR=1
    export RAY_TMPDIR="$ray_tmp"
    export TMPDIR="$ray_tmp"
    export AGL_SERVER_HOST=127.0.0.1
    export AGL_SERVER_PORT="$eval_port"
    export NO_PROXY="$PROXY_NO_PROXY"
    export no_proxy="$PROXY_NO_PROXY"
    export HTTP_PROXY=
    export HTTPS_PROXY=
    export ALL_PROXY=
    export http_proxy=
    export https_proxy=
    export all_proxy=

    cd "$SPIDER_DIR"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_EVAL" python train_sql_agent.py local_qwen05 \
      --config-file "$cfg" \
      --stage 1 \
      --agentlightning-port "$eval_port" \
      --resume-ckpt "$ckpt" >> "$launcher_log" 2>&1
    local rc=$?
    set -e
    echo "$rc" > "$run_root/exit_code.txt"
  )

  local latest_run
  latest_run="$(find "$run_root" -maxdepth 1 -type d -name '*_config_*' | sort | tail -n 1)"
  if [[ -z "$latest_run" ]]; then
    echo -e "${label}\t${split}\tNA\tNA\tno_run_dir\t-" >> "$SUMMARY"
    return
  fi

  local status="ok"
  if [[ -f "$latest_run/error.txt" ]]; then
    status="error"
  elif [[ -f "$latest_run/final_status.json" ]]; then
    status="$(python - <<PY
import json
from pathlib import Path
p = Path("$latest_run/final_status.json")
try:
    print(json.loads(p.read_text(encoding="utf-8")).get("status", "unknown"))
except Exception:
    print("unknown")
PY
)"
  fi

  local success reward
  # Eval metrics are typically emitted in launcher.log as:
  # step:0 - ... - val/n_success:136 - val/reward:0.272 - ...
  success="$(rg -o 'val/n_success:[0-9]+' "$launcher_log" | tail -n1 | cut -d: -f2 || true)"
  reward="$(rg -o 'val/reward:[0-9.eE+-]+' "$launcher_log" | tail -n1 | cut -d: -f2 || true)"
  # Backward-compatible fallback for old progress formats.
  if [[ -z "${success:-}" ]]; then
    success="$(rg -o 'success=[0-9]+' "$latest_run/progress.txt" | tail -n1 | cut -d= -f2 || true)"
  fi
  if [[ -z "${reward:-}" ]]; then
    reward="$(rg -o 'val/reward=[0-9.]+' "$latest_run/progress.txt" | tail -n1 | cut -d= -f2 || true)"
  fi
  success="${success:-NA}"
  reward="${reward:-NA}"

  echo -e "${label}\t${split}\t${success}\t${reward}\t${status}\t${latest_run}" >> "$SUMMARY"
}

GRPO_CKPT="${GRPO_CKPT:-}"
PASSK_CKPT="${PASSK_CKPT:-}"
GUIDED_CKPT="${GUIDED_CKPT:-}"

run_eval "grpo" "$GRPO_CKPT" "dev500" "$CFG_DEV500"
run_eval "passk" "$PASSK_CKPT" "dev500" "$CFG_DEV500"
run_eval "passk_guided" "$GUIDED_CKPT" "dev500" "$CFG_DEV500"

run_eval "grpo" "$GRPO_CKPT" "hard_raw" "$CFG_HARD"
run_eval "passk" "$PASSK_CKPT" "hard_raw" "$CFG_HARD"
run_eval "passk_guided" "$GUIDED_CKPT" "hard_raw" "$CFG_HARD"

if [[ "$RUN_FULL_TEST" == "1" ]]; then
  run_eval "grpo" "$GRPO_CKPT" "test_full" "$CFG_TEST"
  run_eval "passk" "$PASSK_CKPT" "test_full" "$CFG_TEST"
  run_eval "passk_guided" "$GUIDED_CKPT" "test_full" "$CFG_TEST"
fi

echo "evaluation summary: $SUMMARY"
cat "$SUMMARY"
