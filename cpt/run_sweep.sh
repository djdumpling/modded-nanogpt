#!/usr/bin/env bash
# Run the 2x2 condition matrix x N seeds and tee logs.
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC=${NPROC:-8}
STEPS=${STEPS:-1000}
SEEDS=${SEEDS:-"0 1 2"}
EVAL_EVERY=${EVAL_EVERY:-50}
EVAL_BATCHES=${EVAL_BATCHES:-32}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}

mkdir -p cpt/results cpt/logs

for sched in constant_low rewarm_cosine; do
  for replay in 0.00 0.05; do
    for seed in $SEEDS; do
      name="${sched}_replay${replay}_seed${seed}"
      log="cpt/logs/${name}.log"
      out="cpt/results/${name}.json"
      if [[ -f "$out" ]]; then
        echo "[skip] $name (already done)"
        continue
      fi
      echo "[run] $name -> $log"
      .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        cpt/cpt_train.py \
          --schedule "$sched" \
          --replay "$replay" \
          --seed "$seed" \
          --steps "$STEPS" \
          --eval_every "$EVAL_EVERY" \
          --eval_batches "$EVAL_BATCHES" \
          --per_device_bs "$PER_DEVICE_BS" \
          --run_name "$name" \
        2>&1 | tee "$log" | tail -n 40
    done
  done
done

echo "all done."
