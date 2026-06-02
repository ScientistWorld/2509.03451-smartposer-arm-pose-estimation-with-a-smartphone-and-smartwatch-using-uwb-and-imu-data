# Experiment Overview

## pose_tracking_synthetic
Tests the central arm-tracking task with leave-one-synthetic-subject-out cross validation. The experiment measures whether predicted elbow and wrist locations match ground-truth 3-D joint positions and includes an IMU-only comparison to verify that the additional range signal is beneficial.

Validated run `45fc335a-49e`: median elbow+wrist error was 13.66 cm for the proposed result and 14.73 cm for the IMU-only comparison.

## uwb_correction_synthetic
Tests whether noisy phone-watch distance observations are made more accurate before they are used for pose tracking. The experiment measures distance error against ground truth and checks that the corrected stream improves over raw range measurements.

Validated run `45fc335a-49e`: corrected distance MAE was 9.48 cm versus 14.02 cm for raw distance observations.
