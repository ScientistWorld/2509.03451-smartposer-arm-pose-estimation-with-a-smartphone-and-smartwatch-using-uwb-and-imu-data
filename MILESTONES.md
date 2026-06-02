# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 11:54] - core_claim
- Job `1ee26c07-551` completed the corrected 10-fold leave-one-synthetic-subject-out benchmark with the paper-default summed loss.
- SmartPoser improved median elbow+wrist error over IMU-only from 14.76 cm to 13.55 cm and reduced raw UWB MAE from 14.01 cm to 9.41 cm.
- Validation passed, evaluation used saved artifacts, and parameter counts were read from method metadata instead of being hard-coded.

### [2026-06-02 11:22] - method_runs
- Continuation audit found the prior `core_claim` too weak: it used a single synthetic held-out subject, an 8x UWB loss weight rather than the paper's unweighted summed loss, permissive evaluation on missing artifacts, and hard-coded parameter counts.
- Fixed the protocol to run 10 leave-one-synthetic-subject-out folds, restored the paper-default loss weighting, wrote parameter-count metadata from trained models, and made evaluation require both SmartPoser and IMU-only outputs.
- Downgraded to `method_runs` until the corrected benchmark job finishes and produces new scores.

### [2026-06-02 11:15] - method_runs
- Continuation audit reviewed `method/`, `baseline/`, `eval/`, scripts, scoring files, and briefing documents.
- Confirmed the implementation is the paper-identifiable SmartPoser algorithm rather than a surrogate, and that evaluation reads saved artifacts without importing from `method/`.
- Rechecked `https://github.com/FIGLAB/SmartPoser.git`; it still returns `Repository not found`, so the documented synthetic substitute remains necessary.

### [2026-06-02 11:03] - core_claim
- Implemented and ran SmartPoser's two-stage UWB correction plus pose-estimation RNN on the synthetic held-out-subject benchmark.
- Verified the complete UWB+IMU method improves over the IMU-only ablation on median elbow+wrist error and reduces structured raw UWB distance error.
- `python validate.py --compare` passes with evaluation artifacts written to `scoring/scores.json`.

## Stop Justification
- Completed at `core_claim` on the corrected substitute-data benchmark. Higher milestones require the original synchronized UWB/IMU/Kinect data or additional independently meaningful settings; the paper's cited release repository remains unavailable.
