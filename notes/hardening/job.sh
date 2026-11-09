#!/bin/bash
# Hardening job (iteration 11). Every stage is resumable: trainers skip finished
# folds and the evaluators only read models.
#   1. write the visible dev participants (private simulator, public seed 7)
#   2. smoke-test both evaluators on 1-epoch models (fails fast)
#   3. snapshot runs of the reference and the IMU-only baseline; pick training
#      length on the dev leave-one-participant-out scores only
#   4. install the reference submission and freeze the baseline into eval/
#   5. reference scores (scores_train / scores_test / scores)
#   6. variants: seeds, longer/shorter training, wider pose net, window offsets
#   7. shortcuts through the unchanged evaluators
#   8. ranks + validation
set -euo pipefail
cd "$(dirname "$0")/../.."
H=notes/hardening
mkdir -p "$H" tmp/snap tmp/variants tmp/shortcuts tmp/smoke tmp/shortcut_train checkpoints
log() { echo "[$(date +%H:%M:%S)] $*"; }

python - <<'EOF'
import sys, numpy, torch
print({"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()}, flush=True)
EOF

log "1. dev data"
if [ ! -f data/smartposer_dev/SHA256SUMS ]; then
  python eval/test/frozen_simulator.py --write-dev data/smartposer_dev
fi
bash scripts/download.sh

log "2. smoke"
if [ ! -f "$H/smoke_scores_test.json" ]; then
  python method/train_smartposer.py --out-dir tmp/smoke/smartposer_smoke --epochs 1
  python baseline/train_imu_only.py --out-dir tmp/smoke/imu_only_smoke --epochs 1
  python eval/train/evaluate.py --checkpoints-dir tmp/smoke --out "$H/smoke_scores_train.json"
  python eval/test/evaluate.py --checkpoints-dir tmp/smoke --out "$H/smoke_scores_test.json"
fi

log "3. snapshots"
python method/train_smartposer.py --out-dir tmp/snap/smartposer --name smartposer --seed 17 --snapshot-epochs 1 25 50 100 200 300 400
python baseline/train_imu_only.py --out-dir tmp/snap/imu_only --seed 19 --snapshot-epochs 25 50 100 200 300 400
if [ ! -f "$H/snapshots_scores_train.json" ]; then
  python eval/train/evaluate.py --checkpoints-dir tmp/snap --out "$H/snapshots_scores_train.json"
fi
python "$H/select_epoch.py" "$H/snapshots_scores_train.json" > "$H/selection.json"
cat "$H/selection.json"
E=$(python -c "import json;print(json.load(open('$H/selection.json'))['smartposer_epochs'])")
EB=$(python -c "import json;print(json.load(open('$H/selection.json'))['imu_only_epochs'])")
log "selected smartposer epochs=$E imu_only epochs=$EB"

log "4. install reference + fixed baseline"
rm -rf checkpoints/smartposer
cp -r "tmp/snap/smartposer_e$E" checkpoints/smartposer
for side in train test; do
  dest="eval/$side/fixed_models/imu_only"
  rm -rf "$dest"
  mkdir -p "$dest"
  for k in $(seq 0 9); do cp "tmp/snap/imu_only_e$EB/fold_$k/model.pt" "$dest/fold_$k.pt"; done
done

log "5. reference scores"
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh > /dev/null
if [ ! -f "$H/snapshots_scores_test.json" ]; then
  python eval/test/evaluate.py --checkpoints-dir tmp/snap --out "$H/snapshots_scores_test.json" > /dev/null
fi

log "6. variants"
python method/train_smartposer.py --out-dir tmp/variants/smartposer_seed18 --epochs "$E" --seed 18
python method/train_smartposer.py --out-dir tmp/variants/smartposer_seed19 --epochs "$E" --seed 19
python method/train_smartposer.py --out-dir tmp/variants/smartposer_wider --epochs "$E" --seed 17 --pose-hidden 48
python method/train_smartposer.py --out-dir tmp/variants/smartposer_offsets --epochs "$E" --seed 17 --random-offsets
if [ "$E" = "400" ]; then
  python method/train_smartposer.py --out-dir tmp/variants/smartposer_e800 --epochs 800 --seed 17
fi
rm -rf tmp/variants/smartposer && cp -r checkpoints/smartposer tmp/variants/smartposer
for e in 1 25 50 100 200 300 400; do
  if [ "$e" != "$E" ]; then rm -rf "tmp/variants/smartposer_e$e" && cp -r "tmp/snap/smartposer_e$e" "tmp/variants/smartposer_e$e"; fi
done
python eval/train/evaluate.py --checkpoints-dir tmp/variants --out "$H/variants_scores_train.json" > /dev/null
python eval/test/evaluate.py --checkpoints-dir tmp/variants --out "$H/variants_scores_test.json" > /dev/null

log "7. shortcuts"
python method/train_smartposer.py --out-dir tmp/shortcut_train/sc_oversized --epochs "$E" --seed 17 --pose-hidden 64
python "$H/make_shortcuts.py" --reference checkpoints/smartposer --oversized tmp/shortcut_train/sc_oversized --out tmp/shortcuts
python eval/train/evaluate.py --checkpoints-dir tmp/shortcuts --out "$H/shortcuts_scores_train.json" > /dev/null
python eval/test/evaluate.py --checkpoints-dir tmp/shortcuts --out "$H/shortcuts_scores_test.json" > /dev/null

log "8. ranks + validation"
python "$H/rank.py" | tee "$H/ranks.md"
python validate.py --post-design | tee "$H/validate.log" || true
log "done"
