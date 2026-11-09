# Evaluation Targets

All targets are computed by `eval/` from the outputs of the submitted fold models (`checkpoints/<method>/fold_<k>/model.pt`). Ranking uses the hidden slice (`bash scripts/evaluate_test.sh` → `scoring/scores_test.json`). The visible slice (`bash scripts/evaluate_train.sh`, `bash scripts/evaluate.sh`) measures the same metrics on the development participants.

## Primary: pose_tracking_synthetic
Held-out-participant elbow and wrist tracking from phone/watch range and inertial streams.
- **Metric**: `median_elbow_wrist_cm` (lower is better)
- **Supporting metrics**: `median_elbow_cm`, `median_wrist_cm`, `mean_elbow_wrist_cm`, `p90_elbow_wrist_cm`, `high_motion_mean_elbow_wrist_cm` (all lower is better)
- **Constraint (gate)**: at most 92.8k parameters+buffers per fold model, with stored tensors and the whole archive within the stated byte limits. Measured from the file and reported as `param_count_k`; a violation fails the run.
- **Constraint (gate)**: pooled median error below the constant-pose floor `mean_pose`.

## Ablation: pose_tracking_synthetic, range vs. no range
Does the absolute phone-watch range improve pose tracking over inertial sensing alone?
- **Metric**: `median_elbow_wrist_cm` of the submission vs. the frozen `imu_only` row (fixed by `eval/`)

## Secondary: uwb_correction_synthetic
Is the submitted range stream closer to the true phone-watch distance than the raw stream?
- **Metric**: `uwb_error_reduction_pct` (higher is better)
- **Supporting metrics**: `corrected_uwb_p90_cm`, `occluded_uwb_mae_cm` (lower is better), `uwb_correction_win_rate` (higher is better); `corrected_uwb_mae_cm` is reported only
- **Constraint (gate)**: mean range error below the raw stream `raw_uwb` (read by the evaluator from its own inputs)

## Run-level constraints (gates)
- All 10 fold models present, loadable, and returning finite outputs of the required shapes.
- No window's output may depend on the other windows in its batch.
- A failed run is scored at the no-skill floor on every metric of both experiments. A valid run's metrics are clipped at the floor values.
