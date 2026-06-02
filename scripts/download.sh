#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after generation: <50 MB
#
# This paper's public release URL is unavailable, so no external files are
# fetched. The substitute benchmark data is generated into workspace-local
# paths so a fresh clone can recreate it without any pre-placed cache.

set -e

cd "$(dirname "$0")/.."

if [ ! -e data/smartposer_synthetic/train_windows.npz ] || [ ! -e data/smartposer_synthetic/test_windows.npz ]; then
  python data/synthetic_smartposer.py --out data/smartposer_synthetic --subjects 10 --frames 1000 --seed 7
fi
