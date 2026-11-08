#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py reference_seed_19 --seed 19 --epochs 60
