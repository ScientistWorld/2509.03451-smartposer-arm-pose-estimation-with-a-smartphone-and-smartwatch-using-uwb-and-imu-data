# Train/Test Evaluation Design Notes

## Checked

- Read `eval/train/evaluate.py` and `eval/test/evaluate.py` side by side for copy-paste errors, cross-import leakage, hardcoded method defaults, and whether scores are attached to method artifacts.
- Confirmed both slices read `checkpoints/*/predictions.npz` artifacts and group methods from metadata or checkpoint directory names; they do not import or rerun `method/`.
- Confirmed `scripts/evaluate_train.sh` calls `eval/train/evaluate.py` and `scripts/evaluate_test.sh` calls `eval/test/evaluate.py`.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Checked `scoring/reference.json` for paper-agreement metric names or descriptions and added required metric descriptions.

## Changed

- Added explicit `"slice": "train"` and `"slice": "test"` markers to the train/test score outputs so artifact mix-ups are easier to detect.
- Strengthened pose scoring beyond median target-frame error:
  - `p90_elbow_wrist_cm` catches methods that only optimize typical samples.
  - `high_motion_mean_elbow_wrist_cm` probes the highest-motion quartile.
  - `forearm_length_mae_cm` penalizes geometrically implausible elbow/wrist predictions.
- Strengthened UWB correction scoring beyond mean corrected error:
  - `corrected_uwb_p90_cm` catches brittle correction tails.
  - `hard_uwb_improvement_cm` measures improvement on the highest raw-error quartile.
  - `uwb_correction_win_rate` checks that correction improves many samples, not only a few large outliers.
- Updated `scoring/reference.json` with the new metrics, coefficients, and descriptions; removed wording that could imply matching published values is the scoring objective.

## Runtime Concerns For Next Iteration

- The train/test split still slices rolling-window prediction artifacts. Adjacent windows overlap heavily, so the hidden slice is not as independent as a subject-level or fold-level held-out split would be. The added high-motion and tail metrics help, but the runner should watch train/test gaps for suspiciously identical scores.
- `high_motion_mean_elbow_wrist_cm` uses the top quartile within each slice. If a very small prediction artifact is submitted, this metric may be noisy; normal fold outputs should have enough windows.
- The added metrics require only keys already produced by the current SmartPoser and IMU-only artifacts: `pred_joints`, `true_joints`, `arm_span`, and for UWB metrics `features`, `true_distance`, `corrected_distance`. Future methods must keep this artifact contract for the relevant experiment.
