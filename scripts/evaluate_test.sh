#!/bin/bash
# Held-out evaluation slice for final method checks.
# Reads the same prediction artifacts as the train slice and writes scoring/scores_test.json.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring

if [ "$#" -eq 1 ] && [ -d "$1" ]; then
  python eval/test/evaluate.py --predictions "$1/*/predictions.npz"
else
  python eval/test/evaluate.py "$@"
fi
