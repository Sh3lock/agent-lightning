#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO/.artifacts}"
ENV_ACTIVATE="${ENV_ACTIVATE:-}"

GPU_PASSK="${GPU_PASSK:-6}"
GPU_GUIDED="${GPU_GUIDED:-7}"
PASSK_PORT_BASE="${PASSK_PORT_BASE:-13000}"
GUIDED_PORT_BASE="${GUIDED_PORT_BASE:-14000}"

PASSK_CKPT="${PASSK_CKPT:-}"
GUIDED_CKPT="${GUIDED_CKPT:-}"

REPORT_DIR="${REPORT_DIR:-$ARTIFACT_ROOT/eval_reports}"
REPORT_MD="${REPORT_MD:-$REPORT_DIR/eval_status.md}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPORT_DIR/run_$RUN_ID"
mkdir -p "$RUN_DIR"

PASSK_LOG="$RUN_DIR/passk.eval.log"
GUIDED_LOG="$RUN_DIR/guided.eval.log"

log() {
  echo "[$(date '+%F %T')] $*"
}

gpu_line() {
  local gpu_idx="$1"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F',' -v g="$gpu_idx" '{
        idx=$1; gsub(/ /,"",idx);
        if (idx==g) {
          util=$2; mem=$3; total=$4;
          gsub(/ /,"",util); gsub(/ /,"",mem); gsub(/ /,"",total);
          printf("gpu=%s util=%s%% mem=%sMiB/%sMiB\n", idx, util, mem, total);
        }
      }'
}

append_markdown_header_once() {
  if [[ ! -f "$REPORT_MD" ]]; then
    cat > "$REPORT_MD" <<'EOF'
# Spider Eval Auto Report

This file is auto-maintained by `run_eval_dual_gpu_and_report.sh`.
EOF
  fi
}

extract_summary_path() {
  local log_file="$1"
  rg 'evaluation summary:' "$log_file" | tail -n 1 | sed 's/.*evaluation summary: //'
}

append_summary_table() {
  local title="$1"
  local summary_tsv="$2"
  {
    echo ""
    echo "### $title"
    echo ""
    if [[ ! -f "$summary_tsv" ]]; then
      echo "- summary file missing: \`$summary_tsv\`"
      return
    fi
    echo "| label | split | success | reward | status | run_dir |"
    echo "|---|---:|---:|---:|---|---|"
    awk -F'\t' 'NR>1 {printf("| %s | %s | %s | %s | %s | `%s` |\n",$1,$2,$3,$4,$5,$6)}' "$summary_tsv"
  } >> "$REPORT_MD"
}

append_truth_lines() {
  local title="$1"
  local summary_tsv="$2"
  {
    echo ""
    echo "### $title Raw Truth Lines"
    echo ""
  } >> "$REPORT_MD"

  if [[ ! -f "$summary_tsv" ]]; then
    echo "- no summary found" >> "$REPORT_MD"
    return
  fi

  while IFS=$'\t' read -r label split success reward status run_dir; do
    [[ "$label" == "label" ]] && continue
    launcher_path="$run_dir/launcher.log"
    if [[ ! -f "$launcher_path" ]]; then
      launcher_path="$(dirname "$run_dir")/launcher.log"
    fi
    {
      echo "- ${label}/${split} (status=${status}, success=${success}, reward=${reward})"
      if [[ -f "$launcher_path" ]]; then
        echo '```text'
        rg 'step:0 - val/|val/n_success|val/reward|\[val\] step 0|\[val-full\]' "$launcher_path" | tail -n 8 || true
        echo '```'
      elif [[ -f "$run_dir/progress.txt" ]]; then
        echo '```text'
        tail -n 8 "$run_dir/progress.txt"
        echo '```'
      elif [[ -f "$run_dir/error.txt" ]]; then
        echo '```text'
        sed -n '1,40p' "$run_dir/error.txt"
        echo '```'
      else
        echo "  - no launcher/progress/error file found under: \`$run_dir\`"
      fi
    } >> "$REPORT_MD"
  done < "$summary_tsv"
}

launch_eval_job() {
  local label="$1"
  local gpu="$2"
  local passk_ckpt="$3"
  local guided_ckpt="$4"
  local port_base="$5"
  local out_log="$6"

  (
    set -euo pipefail
    if [[ -n "$ENV_ACTIVATE" ]]; then
      source "$ENV_ACTIVATE"
    fi
    cd "$REPO"
    export ARTIFACT_ROOT="$ARTIFACT_ROOT"
    export RAY_TMPDIR_BASE="$ARTIFACT_ROOT/tmp"
    export RAY_LOCAL_FS_CAPACITY_THRESHOLD=0.99
    export RAY_DATA_DISK_USAGE_THRESHOLD=0.99
    export SPIDER_SAVE_RAW_LOG=1
    export SPIDER_RAW_LOG_LEVEL=INFO
    export EVAL_AGENTLIGHTNING_PORT_BASE="$port_base"
    export GRPO_CKPT=""
    export PASSK_CKPT="$passk_ckpt"
    export GUIDED_CKPT="$guided_ckpt"
    GPU_EVAL="$gpu" bash run_spider_config2_eval_matrix.sh
  ) > "$out_log" 2>&1
}

append_markdown_header_once

if [[ -z "$PASSK_CKPT" && -z "$GUIDED_CKPT" ]]; then
  echo "Set PASSK_CKPT and/or GUIDED_CKPT before running this script." >&2
  exit 1
fi

{
  echo ""
  echo "## Run $RUN_ID"
  echo ""
  echo "- start_time: $(date '+%F %T %Z')"
  echo "- gpu_passk: $GPU_PASSK ($(gpu_line "$GPU_PASSK"))"
  echo "- gpu_guided: $GPU_GUIDED ($(gpu_line "$GPU_GUIDED"))"
  echo "- passk_port_base: $PASSK_PORT_BASE"
  echo "- guided_port_base: $GUIDED_PORT_BASE"
  echo "- passk_ckpt: \`$PASSK_CKPT\`"
  echo "- guided_ckpt: \`$GUIDED_CKPT\`"
  echo "- run_dir: \`$RUN_DIR\`"
} >> "$REPORT_MD"

log "Launching PASSK eval on GPU $GPU_PASSK"
launch_eval_job "passk" "$GPU_PASSK" "$PASSK_CKPT" "" "$PASSK_PORT_BASE" "$PASSK_LOG" &
PASSK_PID=$!

sleep 2

log "Launching GUIDED eval on GPU $GPU_GUIDED"
launch_eval_job "guided" "$GPU_GUIDED" "" "$GUIDED_CKPT" "$GUIDED_PORT_BASE" "$GUIDED_LOG" &
GUIDED_PID=$!

set +e
wait "$PASSK_PID"
PASSK_RC=$?
wait "$GUIDED_PID"
GUIDED_RC=$?
set -e

PASSK_SUMMARY="$(extract_summary_path "$PASSK_LOG")"
GUIDED_SUMMARY="$(extract_summary_path "$GUIDED_LOG")"

{
  echo ""
  echo "- end_time: $(date '+%F %T %Z')"
  echo "- passk_rc: $PASSK_RC"
  echo "- guided_rc: $GUIDED_RC"
  echo "- passk_log: \`$PASSK_LOG\`"
  echo "- guided_log: \`$GUIDED_LOG\`"
  echo "- passk_summary: \`${PASSK_SUMMARY:-N/A}\`"
  echo "- guided_summary: \`${GUIDED_SUMMARY:-N/A}\`"
} >> "$REPORT_MD"

append_summary_table "PASSK Summary" "${PASSK_SUMMARY:-}"
append_summary_table "GUIDED Summary" "${GUIDED_SUMMARY:-}"
append_truth_lines "PASSK" "${PASSK_SUMMARY:-}"
append_truth_lines "GUIDED" "${GUIDED_SUMMARY:-}"

log "Done. Report updated at: $REPORT_MD"
log "Run artifacts: $RUN_DIR"
log "passk_rc=$PASSK_RC guided_rc=$GUIDED_RC"
