#!/bin/bash
# Reproduce the reference end to end: verify data, (re)build the fixed IMU-only
# baseline only if it is missing (or REBUILD_BASELINE=1), train SmartPoser on
# the 10 leave-one-participant-out folds, and score the visible and held-out
# slices.

set -e

cd "$(dirname "$0")/.."

bash scripts/download.sh
if [ "${REBUILD_BASELINE:-0}" = "1" ] || [ ! -f eval/train/fixed_models/imu_only/fold_9.pt ] || [ ! -f eval/test/fixed_models/imu_only/fold_9.pt ]; then
  REFREEZE_BASELINE=1 SKIP_EVAL=1 bash scripts/baseline.sh
fi
rm -rf checkpoints/smartposer
SKIP_EVAL=1 bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
