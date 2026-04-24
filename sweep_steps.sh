#!/usr/bin/env bash
# Step-reduction sweep at a fixed rotating_head_dim. For each step count
# in STEPS, patch num_scheduled_iterations in train_gpt.py and run.
# Logs are renamed to rotdim<VAL>_steps<N>__<uuid>.txt.
#
# Use after multi-trial confirmation: pick the rotdim that reliably wins,
# then ask "how few scheduled iterations can we use and still hit 3.28?".
#
# Do NOT launch this while another torchrun is using the GPUs.
set -euo pipefail

ROTDIM=80                         # the rotdim you want to capitalize on
STEPS=(1440 1430 1420 1410 1400)  # values to try for num_scheduled_iterations
TRAIN_FILE="train_gpt.py"
LOG_DIR="logs"

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

cp "$TRAIN_FILE" "${TRAIN_FILE}.sweep.bak"
trap 'mv "${TRAIN_FILE}.sweep.bak" "$TRAIN_FILE"' EXIT

# Lock rotdim once for the whole sweep.
sed -i.tmp -E "s/^([[:space:]]*)rotating_head_dim=[0-9]+,/\1rotating_head_dim=${ROTDIM},/" "$TRAIN_FILE"
rm -f "${TRAIN_FILE}.tmp"
patched_rd=$(grep -E "^[[:space:]]*rotating_head_dim=[0-9]+," "$TRAIN_FILE" || true)
if [[ -z "$patched_rd" ]]; then
  echo "ERROR: sed did not patch rotating_head_dim — aborting." >&2
  exit 1
fi

for N in "${STEPS[@]}"; do
  echo "=========================================="
  echo "  rotating_head_dim=${ROTDIM}  num_scheduled_iterations=${N}"
  echo "=========================================="

  sed -i.tmp -E "s/^([[:space:]]*)num_scheduled_iterations:[[:space:]]*int[[:space:]]*=[[:space:]]*[0-9]+/\1num_scheduled_iterations: int = ${N}/" "$TRAIN_FILE"
  rm -f "${TRAIN_FILE}.tmp"
  patched_st=$(grep -E "^[[:space:]]*num_scheduled_iterations:[[:space:]]*int[[:space:]]*=[[:space:]]*[0-9]+" "$TRAIN_FILE" || true)
  if [[ -z "$patched_st" ]]; then
    echo "ERROR: sed did not patch num_scheduled_iterations — aborting." >&2
    exit 1
  fi
  echo "patched: $patched_rd"
  echo "patched: $patched_st"

  before=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)
  torchrun --standalone --nproc_per_node=8 "$TRAIN_FILE"
  after=$(ls -1 "$LOG_DIR" 2>/dev/null | sort || true)

  new_log=$(comm -13 <(echo "$before") <(echo "$after") | grep '\.txt$' | head -1 || true)
  if [[ -n "$new_log" ]]; then
    mv "$LOG_DIR/$new_log" "$LOG_DIR/rotdim${ROTDIM}_steps${N}__$new_log"
    echo "log -> $LOG_DIR/rotdim${ROTDIM}_steps${N}__$new_log"
  else
    echo "WARN: no new .txt log found for rotdim=${ROTDIM} steps=${N}" >&2
  fi
done

echo "step-reduction sweep done."
