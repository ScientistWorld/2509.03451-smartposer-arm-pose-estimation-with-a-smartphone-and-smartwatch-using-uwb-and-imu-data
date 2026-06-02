#!/bin/bash
# Visible evaluation slice for future method-development iterations.
# Reads pre-generated prediction artifacts and writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 1 ] && [ -d "$1" ]; then
  python eval/train/evaluate.py --predictions "$1/*/predictions.npz"
else
  python eval/train/evaluate.py "$@"
fi
