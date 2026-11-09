#!/bin/bash
# Standard evaluation: leave-one-participant-out scores on the 10 visible
# development participants for every submission under checkpoints/
# (checkpoints/<method>/fold_<k>/model.pt, TorchScript). Uses only eval/ code.
# Writes scoring/scores.json.
#
# Usage:
#   bash scripts/evaluate.sh [--checkpoints-dir DIR]

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/train/evaluate.py --out scoring/scores.json "$@"
