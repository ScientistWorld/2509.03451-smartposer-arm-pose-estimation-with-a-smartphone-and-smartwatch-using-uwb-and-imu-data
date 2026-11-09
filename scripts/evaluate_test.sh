#!/bin/bash
# Held-out slice: every submission under checkpoints/ (all 10 fold models) is
# run on held-out participants generated inside eval/test/ at test time. No
# arguments needed. Writes scoring/scores_test.json.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/test/evaluate.py "$@"
