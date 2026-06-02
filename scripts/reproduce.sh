#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

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
