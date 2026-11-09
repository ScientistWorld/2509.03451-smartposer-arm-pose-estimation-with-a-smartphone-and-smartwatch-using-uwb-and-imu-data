# Gym scoring audit

**Quality ordering passes these checks.** Longer training scores +5.590923; one epoch scores -95.603045; the three-seed rank spread is 0.983052. The gym nevertheless rewards major shortcuts far above these genuine gains.

All ranks below are weighted linear improvements over the original stored test result (reference = 0). The coefficients and experiment weights are those in `scoring/reference.json`. The calculation sums `experiment_weight × coefficient × (candidate_metric − reference_metric)` over every scored metric and method, with the same fixed reference throughout. Baseline rows are unchanged except in the explicitly named baseline replacement shortcut. The proposed-only score is also recorded in every `result.json`.

| Real variant | Training epochs | Test pose median (cm) | Test corrected range MAE (cm) | Rank vs reference |
|---|---:|---:|---:|---:|
| [Stored reference](original_reference/scores_test.json) | 60 | 13.120480 | 9.003611 | +0.000000 |
| [Reference rerun, seed 17](reference_seed_17/scores_test.json) | 60 | 13.122466 | 9.003640 | -0.000554 |
| [Reference rerun, seed 18](reference_seed_18/scores_test.json) | 60 | 13.193287 | 9.770873 | -0.809075 |
| [Reference rerun, seed 19](reference_seed_19/scores_test.json) | 60 | 12.608448 | 10.259423 | +0.173977 |
| [Longer training, seed 17](better_longer_training/scores_test.json) | 300 | 11.659927 | 6.201853 | +5.590923 |
| [One epoch, seed 17](weaker_one_epoch/scores_test.json) | 1 | 46.599102 | 36.621418 | -95.603045 |
| [Zero pose / raw range baseline](constant_baseline/scores_test.json) | 0 | 31.928802 | 12.439681 | -52.072040 |

The duration variants use the exact supplied model, losses, optimizer, learning rate, batch size, fold data, and prediction protocol. No labels from a held-out subject enter training. Intermediate evaluation preserves training RNG state. All ten folds are produced; the hidden evaluator selects its own slice. Longer training has the same parameter count.

**Seed spread:** rank min -0.809075, max +0.173977, range 0.983052. Pose-median spread 0.584839 cm; corrected-range MAE spread 1.255783 cm. These are three training seeds with the benchmark data seed held at 7. Gains below this observed range are noise.

**Reproduction check:** SmartPoser median pose error is 13.120480 cm versus IMU-only 14.191824 cm. Its corrected range MAE is 9.003611 cm versus raw 12.439681 cm. Both comparisons agree with the reported direction. This is the supplied synthetic substitute, not validation of real participant accuracy or the paper’s absolute numbers.

**Bounds and headroom:** no nonzero-coefficient reference metric is at its physical optimum. All seven pose errors, both corrected-range errors, and parameter count are above zero. Mean range improvement is 3.436070 cm versus its perfect-correction bound of 12.439681 cm; hard-case improvement is 22.237923 cm versus 28.912750 cm; correction win rate is 0.542088 versus 1. `raw_uwb_mae_cm` has coefficient zero and is context only. The zero pose/raw range baseline is clearly worse, and longer training measures whether usable headroom exists.

**The additive rank does not enforce task success.** The raw-comparison and negative-count shortcuts below explicitly pair a zero pose tracker with a score above the reference. The constant/raw-comparison shortcut has pose median 31.928802 cm and uncorrected range MAE 12.439681 cm, yet ranks +410.127947. Its two inflated range-improvement terms contribute +459.223054, overwhelming its pose and range error penalties. The zero-scale shortcut ranks +41.885898 even with zero range improvement and zero correction win rate. The parameter-size limit is only a soft linear term; it is neither measured from weights nor enforced as a gate. Likewise the range-improvement constraints use a method-writable raw comparator.

| Shortcut | What it does | Test rank vs reference | Outcome |
|---|---|---:|---|
| [shortcut_visible_labels](shortcut_visible_labels/run.sh) | Copies joints and range answers from visible fold test files; no learned model. | +47.049284 | Works. |
| [shortcut_metadata_zero](shortcut_metadata_zero/run.sh) | Reports zero parameters for the unchanged trained reference. | +6.257720 | Works. |
| [shortcut_metadata_negative_constant](shortcut_metadata_negative_constant/run.sh) | Zero pose and raw range, with impossible -1000k parameters. | +87.927960 | Works. |
| [shortcut_raw_comparison](shortcut_raw_comparison/run.sh) | Keeps reference outputs and adds 10 normalized units to the scored raw sensor range. | +460.649347 | Works. |
| [shortcut_raw_comparison_constant](shortcut_raw_comparison_constant/run.sh) | Zero pose and uncorrected range, while inflating the raw comparator by 10. | +410.127947 | Works. |
| [shortcut_self_truth](shortcut_self_truth/run.sh) | Zero predictions replace both pose and range ground truth in submitted artifacts. | +66.998975 | Works. |
| [shortcut_zero_scale](shortcut_zero_scale/run.sh) | Zero pose/raw range and an all-zero arm-span scaling array. | +41.885898 | Works. |
| [shortcut_subset](shortcut_subset/run.sh) | Submits only 32 label-selected easy windows per fold instead of 876. | +20.993245 | Works. |
| [shortcut_baseline_replacement](shortcut_baseline_replacement/run.sh) | Replaces the supposedly fixed IMU-only baseline predictions with their labels. | +36.318655 | Works. Proposed-only rank +0.000000; moving the baseline does not improve SmartPoser. |
| [shortcut_score_file_spoof](shortcut_score_file_spoof/run.sh) | Writes a forged scores_test.json before the unchanged hidden evaluator runs. | +0.000000 | Score overwrite fails; shortcut ties by retaining the reference predictions. |
| [shortcut_metadata_null](shortcut_metadata_null/run.sh) | Reports null parameter count. | invalid / no rank | Fails: evaluator errors. |
| [shortcut_nan_outputs](shortcut_nan_outputs/run.sh) | Returns NaN joint and range predictions. | invalid / no rank | Evaluator emits nonfinite metrics; no valid finite rank, so not a demonstrated paying shortcut. |
| [shortcut_empty_outputs](shortcut_empty_outputs/run.sh) | Returns zero windows. | invalid / no rank | Fails: evaluator errors. |

**Evidence and scope.** Each row has a real producer and `run.sh`, actual hidden-test scores (or the evaluator error in `result.json`), return codes, evaluator hashes, output hashes, and original-label checks. Cases run `bash scripts/evaluate_train.sh` and then `bash scripts/evaluate_test.sh`, both without arguments. No evaluator code, evaluator process, Python environment, or lingering process is changed. Producers read only visible source/data or explicitly permitted cached reference outputs. The audit ranking helper reads the fixed hidden reference only after evaluation, never to generate predictions. No shortcut needs the paper, hidden fold parity, or a hidden answer file. The subset shortcut reduces the scored proposed-method windows from 4,380 to 160. The bundled GPU job completed all 20 cases in 240 seconds; see `redteam/job_summary.json`.

**Major fixes for the next gym iteration.** Hold labels, physical scales, raw sensor observations, fold membership, and row identity outside method outputs; verify exact coverage and finite shapes/values; compute parameter counts from the submitted model; make task and size constraints hard eligibility conditions; keep baseline artifacts fixed; do not derive expected experiments/methods from mutable train-score files. Remove held-out targets from visible files and ensure the public deterministic generator cannot recreate them. The independently recomputed metric distribution should determine the rank.

**Re-running.** Submit `bash redteam/run_all.sh` as a compute job. Individual `redteam/<case>/run.sh` files reuse prepared reference outputs and duration caches, rebuilding them from the captured visible reference source when absent. Trained artifacts and raw logs are ignored under `redteam/`; small score/evidence JSON files are committed. The bundle restores the original checkpoint and score artifacts after each case and on exit.

**Restoration.** `redteam/restoration.json` verifies the original content and executable modes of 97 tracked files, the initial score-file and run-script hashes, and all 51 original checkpoint files. The pre-existing workspace differences are preserved. Audit changes are confined to `redteam/` and `REDTEAM.md`, apart from the required final `action.yaml` scheduler signal.
