#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py weaker_one_epoch --seed 17 --epochs 1
