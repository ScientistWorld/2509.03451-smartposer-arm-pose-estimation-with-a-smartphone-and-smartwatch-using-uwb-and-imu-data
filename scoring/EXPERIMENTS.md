# Experiment Overview

## pose_tracking_synthetic
Tests the central arm-tracking task with leave-one-synthetic-subject-out cross validation. The experiment measures whether predicted elbow and wrist locations match ground-truth 3-D joint positions, with robustness checks for high-motion windows and implausible forearm lengths.

Validated run `45fc335a-49e`: median elbow+wrist error was 13.66 cm for the reference method and 14.73 cm for the IMU-only comparison.

## uwb_correction_synthetic
Tests whether noisy phone-watch distance observations are made more accurate before they are used for pose tracking. The experiment measures distance error against ground truth and checks that the corrected stream improves over raw range measurements.

Validated run `45fc335a-49e`: corrected distance MAE was 9.48 cm versus 14.02 cm for raw distance observations.

## pose_tracking_gross_synthetic
Tests the arm-tracking task on a wider gross-motor substitute setting. This setting is intended to expose methods that only work on front-of-body or low-variance motions.

Pending scheduler run: `scripts/run.sh` now generates this setting from source and evaluates it alongside the validated combined setting.

## uwb_correction_gross_synthetic
Tests whether noisy phone-watch distance observations are corrected on the wider gross-motor substitute setting.

Pending scheduler run: `scripts/run.sh` now writes this score from generated SmartPoser prediction artifacts.
