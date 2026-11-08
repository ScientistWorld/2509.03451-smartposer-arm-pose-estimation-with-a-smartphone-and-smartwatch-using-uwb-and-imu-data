#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py better_longer_training --seed 17 --epochs 300
