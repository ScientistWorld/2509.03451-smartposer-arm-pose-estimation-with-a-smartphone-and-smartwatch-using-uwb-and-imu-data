#!/bin/bash
# Compute job entry point for evaluator packaging validation.
# The method is frozen; this only evaluates existing prediction artifacts.

set -e

cd "$(dirname "$0")/.."

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh checkpoints
bash scripts/evaluate_test.sh checkpoints
python validate.py --compare
