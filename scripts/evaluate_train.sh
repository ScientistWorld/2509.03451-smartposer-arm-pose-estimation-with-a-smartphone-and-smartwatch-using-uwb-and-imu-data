#!/bin/bash
# Visible slice: leave-one-participant-out scores on the 10 development
# participants for every submission under checkpoints/. No arguments needed.
# Writes scoring/scores_train.json.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/train/evaluate.py "$@"
