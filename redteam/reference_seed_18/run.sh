#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py reference_seed_18 --seed 18 --epochs 60
