#!/bin/bash
# Compute job entry point.
# Regenerates prediction artifacts from source, then evaluates them. Checkpoints
# are gitignored, so this script must not rely on pre-existing local artifacts.

set -e

cd "$(dirname "$0")/.."

bash scripts/reproduce.sh
python validate.py --compare
