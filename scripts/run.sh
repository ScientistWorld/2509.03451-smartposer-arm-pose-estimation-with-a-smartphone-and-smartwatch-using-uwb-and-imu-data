#!/bin/bash
# Compute job entry point.
# Regenerates prediction artifacts from source, then evaluates them. Checkpoints
# are gitignored, so this script must not rely on pre-existing local artifacts.

set -e

cd "$(dirname "$0")/.."

rm -rf data/smartposer_synthetic_fold_* data/smartposer_gross_fold_* checkpoints/imu_only* checkpoints/smartposer* scoring/scores.json

# Core validated setting: combined daily-like and gross-like motions.
SUBJECTS="${SUBJECTS:-10}"
FRAMES="${FRAMES:-1000}"
SEED="${SEED:-7}"
for fold in $(seq 0 $((SUBJECTS - 1))); do
  data_dir="data/smartposer_synthetic_fold_${fold}"
  python data/synthetic_smartposer.py --out "${data_dir}" --subjects "${SUBJECTS}" --frames "${FRAMES}" --seed "${SEED}" --test-subject "${fold}" --profile combined
  SKIP_EVAL=1 bash scripts/baseline.sh --data-dir "${data_dir}" --out-dir "checkpoints/imu_only_fold_${fold}" --epochs "${BASELINE_EPOCHS:-45}" --experiment-name pose_tracking_synthetic
  SKIP_EVAL=1 bash scripts/method.sh --data-dir "${data_dir}" --out-dir "checkpoints/smartposer_fold_${fold}" --epochs "${METHOD_EPOCHS:-60}" --uwb-loss-weight "${UWB_LOSS_WEIGHT:-1.0}" --experiment-name pose_tracking_synthetic
done

# Additional paper-aligned setting: wider gross-motor arm configurations.
GROSS_SUBJECTS="${GROSS_SUBJECTS:-5}"
GROSS_FRAMES="${GROSS_FRAMES:-800}"
for fold in $(seq 0 $((GROSS_SUBJECTS - 1))); do
  data_dir="data/smartposer_gross_fold_${fold}"
  python data/synthetic_smartposer.py --out "${data_dir}" --subjects "${GROSS_SUBJECTS}" --frames "${GROSS_FRAMES}" --seed "${SEED}" --test-subject "${fold}" --profile gross
  SKIP_EVAL=1 bash scripts/baseline.sh --data-dir "${data_dir}" --out-dir "checkpoints/imu_only_gross_fold_${fold}" --epochs "${GROSS_BASELINE_EPOCHS:-40}" --experiment-name pose_tracking_gross_synthetic
  SKIP_EVAL=1 bash scripts/method.sh --data-dir "${data_dir}" --out-dir "checkpoints/smartposer_gross_fold_${fold}" --epochs "${GROSS_METHOD_EPOCHS:-55}" --uwb-loss-weight "${UWB_LOSS_WEIGHT:-1.0}" --experiment-name pose_tracking_gross_synthetic
done

bash scripts/evaluate.sh
python validate.py --compare
