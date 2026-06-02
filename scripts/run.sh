#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

rm -rf data/smartposer_synthetic_fold_* checkpoints/imu_only* checkpoints/smartposer* scoring/scores.json

SUBJECTS="${SUBJECTS:-10}"
FRAMES="${FRAMES:-1000}"
SEED="${SEED:-7}"

for fold in $(seq 0 $((SUBJECTS - 1))); do
  data_dir="data/smartposer_synthetic_fold_${fold}"
  python data/synthetic_smartposer.py --out "${data_dir}" --subjects "${SUBJECTS}" --frames "${FRAMES}" --seed "${SEED}" --test-subject "${fold}"
  SKIP_EVAL=1 bash scripts/baseline.sh --data-dir "${data_dir}" --out-dir "checkpoints/imu_only_fold_${fold}" --epochs "${BASELINE_EPOCHS:-45}"
  SKIP_EVAL=1 bash scripts/method.sh --data-dir "${data_dir}" --out-dir "checkpoints/smartposer_fold_${fold}" --epochs "${METHOD_EPOCHS:-60}" --uwb-loss-weight "${UWB_LOSS_WEIGHT:-1.0}"
done

bash scripts/evaluate.sh
python validate.py --compare
