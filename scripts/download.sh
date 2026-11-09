#!/bin/bash
# Fetch / verify all data and models needed for this environment.
#
# Idempotent; safe to run repeatedly.
#
# Estimated total download size: 0 GB (nothing is fetched from the network).
# Estimated disk usage: ~15 MB (development participants tracked in git).
#
# The paper's public release (https://github.com/FIGLAB/SmartPoser) is not
# reachable, so this benchmark uses a simulated substitute for the
# synchronized UWB/IMU/Kinect recordings. The 10 visible development
# participants are versioned in git under data/smartposer_dev/ (written once
# by the simulator with public seed 7). The simulator itself is private to the
# hidden test, so it can neither mint extra training participants nor
# regenerate the held-out ones. No pretrained weights are used.

set -e

cd "$(dirname "$0")/.."

DEV=data/smartposer_dev
complete=1
[ -f "$DEV/SHA256SUMS" ] || complete=0
for k in 00 01 02 03 04 05 06 07 08 09; do
  [ -f "$DEV/subject_${k}.npz" ] || complete=0
done

if [ "$complete" = 0 ]; then
  if [ -f eval/test/frozen_simulator.py ]; then
    python eval/test/frozen_simulator.py --write-dev "$DEV"
  else
    git checkout -- "$DEV" 2>/dev/null || {
      echo "data/smartposer_dev is tracked in git; restore it with: git checkout -- $DEV" >&2
      exit 1
    }
  fi
fi

(cd "$DEV" && sha256sum --check --quiet SHA256SUMS)
echo "development participants OK: $DEV"
