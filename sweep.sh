#!/usr/bin/env bash
# Sweep rotating_head_dim over a list of values.
# Patches the single hardcoded constant in train_gpt.py, runs training, then
# renames the resulting log so each run is identifiable.
#
# Do NOT launch this while another torchrun is using the GPUs.
set -euo pipefail

VALUES=(32 48 64 80 96)
TRAIN_FILE="train_gpt.py"
LOG_DIR="logs"

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

# Restore the original file on any exit so the working tree is left clean.
cp "$TRAIN_FILE" "${TRAIN_FILE}.sweep.bak"
trap 'mv "${TRAIN_FILE}.sweep.bak" "$TRAIN_FILE"' EXIT

for VAL in "${VALUES[@]}"; do
  echo "=========================================="
  echo "  rotating_head_dim=${VAL}"
  echo "=========================================="

  # Match the unique line `    rotating_head_dim=<int>,` at GPT instantiation.
  # Portable across BSD/GNU sed via the .tmp backup form.
  sed -i.tmp -E "s/^([[:space:]]*)rotating_head_dim=[0-9]+,/\1rotating_head_dim=${VAL},/" "$TRAIN_FILE"
  rm -f "${TRAIN_FILE}.tmp"

  patched=$(grep -E "^[[:space:]]*rotating_head_dim=[0-9]+," "$TRAIN_FILE" || true)
  if [[ -z "$patched" ]]; then
    echo "ERROR: sed did not patch $TRAIN_FILE — aborting."
    exit 1
  fi
  echo "patched: $patched"

  before=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)

  torchrun --standalone --nproc_per_node=8 "$TRAIN_FILE"

  after=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)
  new_log=$(comm -13 <(echo "$before") <(echo "$after") | grep '\.txt$' | head -1 || true)
  if [[ -n "$new_log" ]]; then
    mv "$LOG_DIR/$new_log" "$LOG_DIR/rotdim${VAL}__$new_log"
    echo "log -> $LOG_DIR/rotdim${VAL}__$new_log"
  else
    echo "WARN: no new .txt log found for rotdim=${VAL}"
  fi
done

echo "sweep done."
