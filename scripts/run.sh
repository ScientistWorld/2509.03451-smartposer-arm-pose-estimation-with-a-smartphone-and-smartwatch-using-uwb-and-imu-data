#!/bin/bash
# Compute job entry point for train/test split scoring.
# The method is frozen; this only evaluates existing prediction artifacts.

set -e

cd /home/user

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
