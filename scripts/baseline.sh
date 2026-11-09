#!/bin/bash
# Train the paper's IMU-only ablation (range channel removed) with
# leave-one-participant-out folds into checkpoints/imu_only/, and install it as
# the evaluator-owned fixed baseline (eval/{train,test}/fixed_models/imu_only)
# when that copy is missing or REFREEZE_BASELINE=1. The evaluators always score
# the fixed copy; checkpoints/imu_only itself is a reserved name and is skipped.
#
# Usage:
#   bash scripts/baseline.sh [extra train_imu_only.py args]

set -e

cd "$(dirname "$0")/.."

bash scripts/download.sh
python baseline/train_imu_only.py --data-dir data/smartposer_dev --out-dir checkpoints/imu_only \
  --epochs "${BASELINE_EPOCHS:-800}" --seed 19 "$@"
for side in train test; do
  dest="eval/$side/fixed_models/imu_only"
  if [ "${REFREEZE_BASELINE:-0}" = "1" ] || [ ! -f "$dest/fold_9.pt" ]; then
    mkdir -p "$dest"
    for k in $(seq 0 9); do
      cp "checkpoints/imu_only/fold_$k/model.pt" "$dest/fold_$k.pt"
    done
  fi
done
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  bash scripts/evaluate.sh
fi
