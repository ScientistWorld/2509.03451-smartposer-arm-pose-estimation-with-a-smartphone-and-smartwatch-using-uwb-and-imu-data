# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The main metric is median positional error in centimeters for elbow and wrist joints, computed as Euclidean distance between predicted and ground-truth 3-D joint coordinates after scaling normalized coordinates by arm span. Lower is better. Supporting metrics include per-joint median error, mean elbow+wrist error, and mean absolute error for range correction.

## Baselines and Targets

The paper reports 11.0 cm median elbow+wrist error, with 8.7 cm elbow and 13.3 cm wrist median error. The IMU-only ablation reports 14.2 cm median elbow+wrist error, with 11.7 cm elbow and 16.7 cm wrist. For range correction, the paper reports raw UWB mean error of 21.0 cm and corrected UWB mean error of 8.6 cm.

## Evaluation Protocol

The original study uses leave-one-participant-out cross validation over 10 participants and reports results across daily activities plus gross motor arm motions. This environment uses a 10-fold leave-one-synthetic-subject-out substitute protocol because the paper's cited repository is unavailable. Evaluation reads prediction artifacts from `checkpoints/*/predictions.npz` and method metadata from `checkpoints/*/metadata.json`; it writes `scoring/scores.json` and does not import from `method/`.
