# Experiment Overview

## pose_tracking_synthetic
Tests the central arm-tracking task on a held-out synthetic subject. The experiment measures whether predicted elbow and wrist locations match ground-truth 3-D joint positions and includes an IMU-only comparison to verify that the additional range signal is beneficial.

## uwb_correction_synthetic
Tests whether noisy phone-watch distance observations are made more accurate before they are used for pose tracking. The experiment measures distance error against ground truth and checks that the corrected stream improves over raw range measurements.
