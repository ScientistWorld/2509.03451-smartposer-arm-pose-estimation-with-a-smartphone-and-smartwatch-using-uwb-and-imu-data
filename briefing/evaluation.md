# Evaluation

## What a submission is

A submission is a directory `checkpoints/<method>/` holding ten TorchScript models,
`fold_0/model.pt` … `fold_9/model.pt`, one per leave-one-participant-out fold of the
ten visible development participants in `data/smartposer_dev/`. The fold-`k` model
must not be trained on development participant `k`.

Each model maps a batch of 125-frame sensor windows to a range stream and a joint
stream:

```
forward(features: float32 [B, 125, 25]) -> (range [B, 125, 1], joints [B, 125, 9])
```

`features` uses the layout of `data/smartposer_dev/meta.json` (raw phone-watch range
divided by arm span, phone orientation, watch orientation in the phone frame, phone and
watch accelerations divided by 30). `range` is the phone-watch distance divided by arm
span; `joints` are shoulder, elbow and wrist xyz in the shoulder frame divided by arm
span. Only output frame 119 (the 120th of 125, i.e. 200 ms of look-ahead) is scored,
and it is compared with the ground truth at that same frame. The evaluator builds the
windows itself from continuous streams (rolling, every 5th frame), shuffles them across
participants and time, and runs each batch through a freshly loaded model. Everything
the scores use (windows, labels, arm spans, the raw range stream, the masks for hard
cases, the baselines) comes from the evaluator, not from the submission. It never reads
metadata or any number a submission writes.

Every directory under `checkpoints/` whose name does not start with `_` is treated as a
submission. Put scratch material in `_`-prefixed directories. `mean_pose`, `raw_uwb`
and `imu_only` are reserved for the fixed baselines and are skipped.

## Slices

- **Visible**: `bash scripts/evaluate_train.sh` writes `scoring/scores_train.json`, and
  `bash scripts/evaluate.sh` writes the same scores to `scoring/scores.json`. The fold-`k`
  model is scored on development participant `k`, and results are pooled over the ten
  participants.
- **Hidden**: `bash scripts/evaluate_test.sh` writes `scoring/scores_test.json`. All ten
  fold models are scored on each of 48 unseen participants (2 minutes each, half
  everyday activity, half wide gross-motor movement). The evaluator generates these
  participants at test time from the same distribution as the development data, and
  results are pooled over models and participants. All three scripts take no arguments
  and score every submission on disk. Ranking uses the hidden slice.

## Gates

If any gate fails, the submission is scored as failed and gets the no-skill floor (the
`mean_pose` row and the `raw_uwb` row) on every metric of both experiments:

- All ten fold models are present, load as TorchScript, and return finite tensors of the
  shapes above for every window.
- **Size.** Each fold model has at most 92,800 parameters and buffers, which is twice the
  paper's 46.4k-parameter model. Its stored tensors take at most 742,400 bytes, and the
  whole archive (code, pickled attributes, extra files) is at most about 1 MB
  uncompressed. All of this is measured from the file.
- **Batch independence.** A window's output must not depend on which other windows share
  its batch. The evaluator re-runs some windows in a different batch, and the outputs
  must match within 1e-3 arm spans.
- **Task success.** Pooled median elbow/wrist error must be below the best constant pose
  (`mean_pose`), and mean range error must be below the raw range stream (`raw_uwb`).

For a valid submission, each scored metric is clipped at the floor value, so a failed
submission always holds the worst value of every metric.

## Experiments and metrics

`pose_tracking_synthetic` (weight 0.7). Errors are Euclidean, in cm (normalized
coordinates × arm span), at the scored frame, and lower is better:
`median_elbow_wrist_cm` (primary), `median_elbow_cm`, `median_wrist_cm`,
`mean_elbow_wrist_cm`, `p90_elbow_wrist_cm`, and `high_motion_mean_elbow_wrist_cm` (the
quartile of frames with the largest true 1-second wrist displacement). `param_count_k`
is reported but not scored, because size is a gate.

`uwb_correction_synthetic` (weight 0.3) compares the submitted range stream against the
true phone-watch distance:

- `uwb_error_reduction_pct` (primary, higher is better): share of the raw stream's mean
  absolute error removed.
- `corrected_uwb_p90_cm` (lower is better).
- `occluded_uwb_mae_cm` (lower is better): the quartile of frames where the raw stream is
  worst.
- `uwb_correction_win_rate` (higher is better): fraction of frames where the submitted
  range is closer to the truth than the raw stream.
- `corrected_uwb_mae_cm` is reported only. On the fixed test it is an affine function of
  the primary metric.

Coefficients are in `scoring/reference.json`.

## Fixed comparisons

- `mean_pose`: the best constant pose, which is the no-skill pose floor.
- `raw_uwb`: the uncorrected range stream, which is the no-skill range floor.
- `imu_only`: frozen models in `eval/*/fixed_models/imu_only/`. They estimate pose from
  the inertial streams alone, with the range channel removed, and are scored on the pose
  experiment only.

## Paper's reported numbers (real recordings, not targets)

On its own 10-participant recordings, the paper reports:

- 11.0 cm median elbow+wrist error (8.7 cm elbow, 13.3 cm wrist).
- 14.2 cm for the inertial-only ablation.
- Raw range error reduced from 21.0 cm to 8.6 cm, a 59% reduction.

The paper's data release is unreachable, so this benchmark uses a simulated substitute
with the same sensors, frame rates, normalization and protocol. Absolute numbers here
are on the substitute's scale and are not meant to match the paper's.
