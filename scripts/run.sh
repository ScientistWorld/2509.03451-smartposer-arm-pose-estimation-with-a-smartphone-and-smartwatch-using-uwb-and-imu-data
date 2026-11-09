#!/bin/bash
# Compute job entry point: reproduce the reference end to end (verify the dev
# data, keep the frozen IMU-only baseline, train SmartPoser on the 10
# leave-one-participant-out folds, score the visible and hidden slices), then
# validate the workspace. The final hardening audit job is notes/hardening/job_v3.sh.
set -e
cd "$(dirname "$0")/.."
bash scripts/reproduce.sh
python validate.py
