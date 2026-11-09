#!/bin/bash
# Final hardening job (v3). Only the evaluators changed since v2 (floor
# clipping, re-batching probe, every checkpoints/ directory is a submission,
# fp32 without TF32), so every model saved by v2 is rescored, not retrained.
# Three new trainings: a same-seed rerun of the reference (reproducibility),
# a random-window-offset variant (candidate better method) and the
# all-participants protocol probe. Every step skips finished outputs.
#   1. dev data + smoke test of both evaluators (fails fast)
#   2. reference scores -> scoring/scores_{train,test}.json, scoring/scores.json
#   3. shortcuts (v2 set + v3 probes) through both evaluators
#   4. new trainings
#   5. variants (v2 saved models + new) through both evaluators
#   6. paired participant bootstrap on the hidden slice
#   7. ranks + validation
set -euo pipefail
cd "$(dirname "$0")/../.."
H=notes/hardening
O=$H/v3
V2=tmp/h2
T=tmp/h3
mkdir -p "$O" "$T/variants" "$T/shortcuts" "$T/smoke"
log() { echo "[$(date +%H:%M:%S)] $*"; }

python - <<'EOF'
import sys, numpy, torch
print({"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()}, flush=True)
EOF

log "1. dev data + smoke"
bash scripts/download.sh
if [ ! -f "$O/smoke_scores_test.json" ]; then
  rm -rf "$T/smoke" && mkdir -p "$T/smoke"
  ln -s "$(pwd)/$V2/smoke/smartposer_smoke" "$T/smoke/smartposer_smoke"
  ln -s "$(pwd)/checkpoints/smartposer" "$T/smoke/smartposer"
  python eval/train/evaluate.py --checkpoints-dir "$T/smoke" --out "$O/smoke_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/smoke" --out "$O/smoke_scores_test.json" > /dev/null
  python "$H/show_status.py" "$O/smoke_scores_train.json" "$O/smoke_scores_test.json"
fi

log "2. reference scores"
ls checkpoints
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null
bash scripts/evaluate.sh > /dev/null
python "$H/show_status.py" scoring/scores_train.json scoring/scores_test.json

log "3. shortcuts"
if [ ! -f "$O/shortcuts_scores_test.json" ]; then
  rm -rf "$T/shortcuts" && mkdir -p "$T/shortcuts"
  for d in "$V2"/shortcuts/*; do ln -s "$(pwd)/$d" "$T/shortcuts/$(basename "$d")"; done
  python "$H/make_shortcuts_v3.py" --reference checkpoints/smartposer --out "$T/shortcuts"
  python eval/train/evaluate.py --checkpoints-dir "$T/shortcuts" --out "$O/shortcuts_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/shortcuts" --out "$O/shortcuts_scores_test.json" > /dev/null
fi
python "$H/show_status.py" "$O/shortcuts_scores_train.json" "$O/shortcuts_scores_test.json"

log "4. new trainings"
python method/train_smartposer.py --data-dir data/smartposer_dev --out-dir "$T/new/smartposer_rerun17" --epochs 800 --seed 17
python method/train_smartposer.py --data-dir data/smartposer_dev --out-dir "$T/new/smartposer_randoffsets" --epochs 800 --seed 17 --random-offsets
python "$H/train_all_participants.py" --out-dir "$T/new_probe/sc_all_participants" --epochs 800 --seed 17

log "5. variants"
if [ ! -f "$O/variants_scores_test.json" ]; then
  rm -rf "$T/variants" && mkdir -p "$T/variants"
  for d in "$V2"/variants/* "$T"/new/*; do ln -s "$(pwd)/$d" "$T/variants/$(basename "$d")"; done
  python eval/train/evaluate.py --checkpoints-dir "$T/variants" --out "$O/variants_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/variants" --out "$O/variants_scores_test.json" > /dev/null
fi
if [ ! -f "$O/protocol_scores_test.json" ]; then
  rm -rf "$T/protocol" && mkdir -p "$T/protocol"
  ln -s "$(pwd)/$T/new_probe/sc_all_participants" "$T/protocol/sc_all_participants"
  python eval/train/evaluate.py --checkpoints-dir "$T/protocol" --out "$O/protocol_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/protocol" --out "$O/protocol_scores_test.json" > /dev/null
fi

log "6. paired noise"
if [ ! -f "$O/paired_noise.json" ]; then
  rm -rf "$T/noise" && mkdir -p "$T/noise"
  for d in "$T"/variants/*; do ln -s "$(readlink -f "$d")" "$T/noise/$(basename "$d")"; done
  ln -s "$(pwd)/$T/new_probe/sc_all_participants" "$T/noise/sc_all_participants"
  python "$H/paired_noise.py" --variants "$T/noise" --reference smartposer --out "$O/paired_noise.json" > "$O/paired_noise.md" \
    || log "step 6 failed (continuing)"
fi
cat "$O/paired_noise.md" || true

log "7. ranks + validation"
python "$H/rank.py" "$O" | tee "$O/ranks.md"
python validate.py | tee "$O/validate.log" || true
log "done"
