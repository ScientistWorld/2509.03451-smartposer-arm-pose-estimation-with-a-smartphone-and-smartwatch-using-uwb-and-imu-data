#!/bin/bash
# Visible evaluation slice for future method-development iterations.
# Reads pre-generated prediction artifacts and writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py "$@"
