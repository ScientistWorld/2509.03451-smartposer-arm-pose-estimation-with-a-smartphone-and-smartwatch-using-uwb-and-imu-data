# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-02 11:44] - core_claim
- Implemented SmartPoser's actual two-stage algorithm: UWB correction BiLSTM followed by pose-estimation BiLSTM trained with summed UWB MSE and MPJPE losses.
- Packaged a method-agnostic synthetic substitute benchmark because the paper's cited repository is unavailable.
- Completed a 10-fold leave-one-synthetic-subject-out scheduler run that passed `python validate.py --compare`.
- Reproduced the central effect on the substitute benchmark: SmartPoser improved median elbow+wrist error over IMU-only and reduced structured raw UWB range error.

### [2026-06-02 11:58] - method_runs
- Continuation audit reset the milestone to `method_runs`.
- Fixed a protocol mismatch in the substitute data: test windows now use a one-frame rolling stride, matching the paper's n-5 rolling-buffer evaluation.
- Fixed synthetic sensor preprocessing to mimic paper inputs more closely: raw UWB is sampled at 5 Hz then upsampled to 25 Hz, and watch orientation/acceleration are expressed relative to the phone frame.
- A new scheduler run is required before restoring `core_claim` with the corrected benchmark.

## Current Limitation
- Higher milestones require the original synchronized UWB/IMU/Kinect release or additional substitute benchmark settings beyond the current validated package.
