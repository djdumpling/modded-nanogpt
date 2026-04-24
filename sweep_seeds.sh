#!/usr/bin/env bash
# Multi-trial confirmation sweep. For each (rotating_head_dim, trial)
# pair, patch train_gpt.py and run training. Logs are renamed to
# rotdim<VAL>_trial<N>__<uuid>.txt so trials are easy to group.
#
# Note: modded-nanogpt sets no explicit RNG seed — non-determinism comes
# from CUDA/NCCL ordering plus a fresh uuid run_id. "Trial N" therefore
# means "Nth re-run of the same config", which is enough to estimate
# run-to-run noise.
#
# Do NOT launch this while another torchrun is using the GPUs.
set -euo pipefail

VALUES=(64 80)        # rotdims to compare
TRIALS=4              # runs per value
TRAIN_FILE="train_gpt.py"
LOG_DIR="logs"

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

cp "$TRAIN_FILE" "${TRAIN_FILE}.sweep.bak"
trap 'mv "${TRAIN_FILE}.sweep.bak" "$TRAIN_FILE"' EXIT

for VAL in "${VALUES[@]}"; do
  sed -i.tmp -E "s/^([[:space:]]*)rotating_head_dim=[0-9]+,/\1rotating_head_dim=${VAL},/" "$TRAIN_FILE"
  rm -f "${TRAIN_FILE}.tmp"

  patched=$(grep -E "^[[:space:]]*rotating_head_dim=[0-9]+," "$TRAIN_FILE" || true)
  if [[ -z "$patched" ]]; then
    echo "ERROR: sed did not patch $TRAIN_FILE — aborting." >&2
    exit 1
  fi

  for TRIAL in $(seq 1 "$TRIALS"); do
    echo "=========================================="
    echo "  rotating_head_dim=${VAL}  trial=${TRIAL}/${TRIALS}"
    echo "=========================================="
    echo "patched: $patched"

    before=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)
    torchrun --standalone --nproc_per_node=8 "$TRAIN_FILE"
    after=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)

    new_log=$(comm -13 <(echo "$before") <(echo "$after") | grep '\.txt$' | head -1 || true)
    if [[ -n "$new_log" ]]; then
      mv "$LOG_DIR/$new_log" "$LOG_DIR/rotdim${VAL}_trial${TRIAL}__$new_log"
      echo "log -> $LOG_DIR/rotdim${VAL}_trial${TRIAL}__$new_log"
    else
      echo "WARN: no new .txt log found for rotdim=${VAL} trial=${TRIAL}" >&2
    fi
  done
done

echo "multi-trial sweep done."
