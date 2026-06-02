#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

if [ ! -e data/smartposer_synthetic/train_windows.npz ]; then
  bash scripts/download.sh
fi

python method/train_smartposer.py "$@"
bash scripts/evaluate.sh
