#!/bin/bash
# Compute job entry point for train/test split scoring.
# The method is frozen; this only evaluates existing prediction artifacts.

set -e

cd "$(dirname "$0")/.."

bash scripts/evaluate_train.sh checkpoints
bash scripts/evaluate_test.sh checkpoints
