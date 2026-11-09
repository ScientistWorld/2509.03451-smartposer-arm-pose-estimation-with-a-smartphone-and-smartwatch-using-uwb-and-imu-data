#!/bin/bash
# Run the paper's proposed method (SmartPoser) with leave-one-participant-out
# training on the 10 development participants. Writes one TorchScript model per
# fold to checkpoints/smartposer/fold_<k>/model.pt, then evaluates.
#
# Usage:
#   bash scripts/method.sh [extra train_smartposer.py args]

set -e

cd "$(dirname "$0")/.."

bash scripts/download.sh
python method/train_smartposer.py --data-dir data/smartposer_dev --out-dir checkpoints/smartposer \
  --epochs "${METHOD_EPOCHS:-200}" --seed "${METHOD_SEED:-17}" "$@"
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  bash scripts/evaluate.sh
fi
