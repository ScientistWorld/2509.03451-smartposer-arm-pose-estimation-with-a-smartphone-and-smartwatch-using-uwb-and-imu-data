#!/bin/bash
set -euo pipefail
cd /home/user
# One bundle, no installs, no network, and no evaluator/environment modifications.
trap 'python redteam/support/restore.py' EXIT
python -u redteam/support/run_case.py original_reference
python -u redteam/support/train_variants.py --seed 17 --epochs 1 60 300
bash redteam/reference_seed_17/run.sh
bash redteam/weaker_one_epoch/run.sh
bash redteam/better_longer_training/run.sh
python -u redteam/support/train_variants.py --seed 18 --epochs 60
bash redteam/reference_seed_18/run.sh
python -u redteam/support/train_variants.py --seed 19 --epochs 60
bash redteam/reference_seed_19/run.sh
bash redteam/constant_baseline/run.sh
for name in shortcut_visible_labels shortcut_metadata_zero shortcut_metadata_negative_constant shortcut_raw_comparison shortcut_raw_comparison_constant shortcut_self_truth shortcut_zero_scale shortcut_subset shortcut_baseline_replacement shortcut_score_file_spoof shortcut_metadata_null shortcut_nan_outputs shortcut_empty_outputs; do
  bash "redteam/${name}/run.sh"
done
python redteam/support/summarize.py
