#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd "$(dirname "$0")/.."

if [ ! -e data/smartposer_synthetic/train_windows.npz ]; then
  bash scripts/download.sh
fi

python method/train_smartposer.py "$@"
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  bash scripts/evaluate.sh
fi
