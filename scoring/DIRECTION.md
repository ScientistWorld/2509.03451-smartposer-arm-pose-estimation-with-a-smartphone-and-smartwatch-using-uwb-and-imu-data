# Research Direction

## Research Problem
Improve mobile arm pose estimation from sparse commodity-device sensors. The benchmark asks whether elbow and wrist position can be recovered from a phone/watch sensor stream without cameras or external infrastructure.

## Core Contribution
The paper's contribution is sensor fusion for wearable pose tracking, especially using absolute phone-watch range to stabilize inertial pose estimation. Future work should improve this class of range-plus-IMU temporal pose estimation.

## Approach Scope
In-bounds approaches may change temporal models, filtering, calibration, feature normalization, losses, range correction, or the way range and inertial streams are fused, as long as the solution remains a small sensor-fusion arm-pose tracker using the provided inputs.

## Out of Bounds
Do not switch to camera-based pose estimation, external motion capture, added body sensors, direct inverse-kinematics access to ground-truth joints, or large generic sequence models that overwhelm the paper's small commodity-device regime. The latter is enforced by the evaluator's per-fold size gate (92.8k parameters+buffers, about 1 MB archive). Ensembles and other composites count as a single model under that gate. Do not pool information across the windows of a batch, and do not train a fold on its own held-out participant.
