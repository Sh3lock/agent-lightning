#!/bin/bash
# Delete recent /tmp entries owned by a user (default: wenbinxing) matching patterns.

set -euo pipefail

# ===== User Config =====
TARGET_DIR="/tmp"
USER_NAME="wenbinxing"
DAYS=5               # "recent 5 days" (mtime)
DRY_RUN=1            # 1 = preview only, 0 = delete

# Only delete names that match these patterns at /tmp top-level
PATTERNS=(
  "tmp*"
  "pymp-*"
  "pyright-*"
  "torchelastic_*"
  "wandb-*"
  "torchinductor_*"
)

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Target dir not found: ${TARGET_DIR}"
  exit 1
fi

if [[ "${#PATTERNS[@]}" -eq 0 ]]; then
  echo "No patterns configured. Refusing to run."
  exit 1
fi

echo "Target: ${TARGET_DIR}"
echo "User: ${USER_NAME}"
echo "Recent: ${DAYS} days (mtime)"
echo "Patterns: ${PATTERNS[*]}"
echo "Dry run: ${DRY_RUN}"
echo ""

find_expr=( -maxdepth 1 -user "${USER_NAME}" -mtime "-${DAYS}" )
name_expr=( -false )
for pat in "${PATTERNS[@]}"; do
  name_expr+=( -o -name "${pat}" )
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Preview (no deletion):"
  find "${TARGET_DIR}" "${find_expr[@]}" \( "${name_expr[@]}" \) -print
  echo ""
  echo "Set DRY_RUN=0 to delete."
  exit 0
fi

echo "Deleting..."
find "${TARGET_DIR}" "${find_expr[@]}" \( "${name_expr[@]}" \) -print0 | xargs -0 rm -rf --
echo "Done."
