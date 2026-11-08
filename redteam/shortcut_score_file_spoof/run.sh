#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -u redteam/support/run_case.py shortcut_score_file_spoof 
