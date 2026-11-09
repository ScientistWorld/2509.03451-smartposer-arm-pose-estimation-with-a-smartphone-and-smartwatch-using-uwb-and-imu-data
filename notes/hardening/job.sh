#!/bin/bash
# Hardening job v2 (iteration 11). Every stage is resumable: trainers skip
# finished folds, evaluators only read models, guarded steps skip saved outputs.
#   1. verify the visible dev participants
#   2. smoke-test both evaluators on 1-epoch models (fails fast; writes the
#      hidden-test fingerprint for the 48-participant test)
#   3. snapshot runs of the reference and the IMU-only baseline (independent
#      seed per fold); pick training length on dev leave-one-participant-out only
#   4. install the reference submission and freeze the baseline into eval/
#   5. reference scores (scores_train / scores_test / scores)
#   6. variants: seeds, 2x longer training, wider pose net, 2-seed ensemble,
#      shorter-training snapshots
#   7. shortcuts through the final evaluators
#   8. paired participant-bootstrap noise on the hidden slice
#   9. ranks + validation
set -euo pipefail
cd "$(dirname "$0")/../.."
H=notes/hardening
T=tmp/h2
mkdir -p "$H" "$T/snap" "$T/variants" "$T/shortcuts" "$T/smoke" "$T/shortcut_train" checkpoints
log() { echo "[$(date +%H:%M:%S)] $*"; }

python - <<'EOF'
import sys, numpy, torch
print({"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()}, flush=True)
EOF

log "1. dev data"
bash scripts/download.sh

log "2. smoke"
if [ ! -f "$H/smoke_scores_test.json" ]; then
  python method/train_smartposer.py --out-dir "$T/smoke/smartposer_smoke" --epochs 1
  python baseline/train_imu_only.py --out-dir "$T/smoke/imu_only_smoke" --epochs 1
  python eval/train/evaluate.py --checkpoints-dir "$T/smoke" --out "$H/smoke_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/smoke" --out "$H/smoke_scores_test.json" > /dev/null
fi

log "3. snapshots"
python method/train_smartposer.py --out-dir "$T/snap/smartposer" --name smartposer --seed 17 --snapshot-epochs 1 25 50 100 200 400 600 800
python baseline/train_imu_only.py --out-dir "$T/snap/imu_only" --seed 19 --snapshot-epochs 25 50 100 200 400 600 800
if [ ! -f "$H/snapshots_scores_train.json" ]; then
  python eval/train/evaluate.py --checkpoints-dir "$T/snap" --out "$H/snapshots_scores_train.json" > /dev/null
fi
python "$H/select_epoch.py" "$H/snapshots_scores_train.json" > "$H/selection.json"
cat "$H/selection.json"
E=$(python -c "import json;print(json.load(open('$H/selection.json'))['smartposer_epochs'])")
EB=$(python -c "import json;print(json.load(open('$H/selection.json'))['imu_only_epochs'])")
log "selected smartposer epochs=$E imu_only epochs=$EB"

log "4. install reference + fixed baseline"
rm -rf checkpoints/smartposer
cp -r "$T/snap/smartposer_e$E" checkpoints/smartposer
for side in train test; do
  dest="eval/$side/fixed_models/imu_only"
  rm -rf "$dest"
  mkdir -p "$dest"
  for k in $(seq 0 9); do cp "$T/snap/imu_only_e$EB/fold_$k/model.pt" "$dest/fold_$k.pt"; done
done

log "5. reference scores"
bash scripts/evaluate_train.sh > /dev/null
bash scripts/evaluate_test.sh > /dev/null
bash scripts/evaluate.sh > /dev/null
if [ ! -f "$H/snapshots_scores_test.json" ]; then
  python eval/test/evaluate.py --checkpoints-dir "$T/snap" --out "$H/snapshots_scores_test.json" > /dev/null
fi

log "6. variants"
python method/train_smartposer.py --out-dir "$T/variants/smartposer_seed18" --epochs "$E" --seed 18
python method/train_smartposer.py --out-dir "$T/variants/smartposer_seed19" --epochs "$E" --seed 19
python "$H/make_variants.py" ens "$T/variants/smartposer_seed18" "$T/variants/smartposer_seed19" "$T/variants/smartposer_ens_seed18_19"
python method/train_smartposer.py --out-dir "$T/variants/smartposer_wider" --epochs "$E" --seed 17 --pose-hidden 48
python method/train_smartposer.py --out-dir "$T/variants/smartposer_x2epochs" --epochs "$((2 * E))" --seed 17
rm -rf "$T/variants/smartposer" && cp -r checkpoints/smartposer "$T/variants/smartposer"
for e in 1 25 50 100 200 400 600 800; do
  if [ "$e" -lt "$E" ]; then rm -rf "$T/variants/smartposer_e$e" && cp -r "$T/snap/smartposer_e$e" "$T/variants/smartposer_e$e"; fi
done
python eval/train/evaluate.py --checkpoints-dir "$T/variants" --out "$H/variants_scores_train.json" > /dev/null
python eval/test/evaluate.py --checkpoints-dir "$T/variants" --out "$H/variants_scores_test.json" > /dev/null

log "7. shortcuts"
shortcuts() {
  python method/train_smartposer.py --out-dir "$T/shortcut_train/sc_oversized" --epochs 25 --seed 17 --pose-hidden 64
  python "$H/make_shortcuts.py" --reference checkpoints/smartposer --oversized "$T/shortcut_train/sc_oversized" --out "$T/shortcuts"
  python eval/train/evaluate.py --checkpoints-dir "$T/shortcuts" --out "$H/shortcuts_scores_train.json" > /dev/null
  python eval/test/evaluate.py --checkpoints-dir "$T/shortcuts" --out "$H/shortcuts_scores_test.json" > /dev/null
  python "$H/show_status.py" "$H/shortcuts_scores_train.json" "$H/shortcuts_scores_test.json"
}
shortcuts || log "step 7 failed (continuing)"

log "8. paired noise"
if [ ! -f "$H/paired_noise.json" ]; then
  python "$H/paired_noise.py" --variants "$T/variants" --reference smartposer --out "$H/paired_noise.json" > "$H/paired_noise.md" \
    || log "step 8 failed (continuing)"
  cat "$H/paired_noise.md" || true
fi

log "9. ranks + validation"
python "$H/rank.py" | tee "$H/ranks.md"
python validate.py --post-design | tee "$H/validate.log" || true
log "done"
