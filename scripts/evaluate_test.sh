#!/bin/bash
# Held-out evaluation slice for final method checks.
# Reads the same prediction artifacts as the train slice and writes scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py "$@"
