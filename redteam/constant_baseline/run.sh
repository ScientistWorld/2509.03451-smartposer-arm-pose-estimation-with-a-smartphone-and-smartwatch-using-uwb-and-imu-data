#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py constant_baseline 
