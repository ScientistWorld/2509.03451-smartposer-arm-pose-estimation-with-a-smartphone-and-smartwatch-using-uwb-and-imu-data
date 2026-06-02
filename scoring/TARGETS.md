# Evaluation Targets

## Primary: pose_tracking_synthetic
Tests whether a system can estimate held-out-user elbow and wrist position from wearable/mobile sensor streams.
- **Metric**: `median_elbow_wrist_cm` (lower is better)
- **Constraint metric**: `param_count_k` (lower is better; prevents trivial scaling)
- **Run**: `bash scripts/evaluate.sh`

## Ablation: pose_tracking_synthetic
Tests whether the complete sensor setup improves over the IMU-only condition.
- **Metric**: `median_elbow_wrist_cm` gap between `smartposer` and `imu_only` (lower SmartPoser error is better)
- **Run**: `bash scripts/evaluate.sh`

## Primary: uwb_correction_synthetic
Tests whether raw range measurements are corrected toward ground-truth phone-watch distance.
- **Metric**: `corrected_uwb_mae_cm` (lower is better)
- **Constraint metric**: `uwb_improvement_cm` (higher is better; correction should improve over raw range)
- **Run**: `bash scripts/evaluate.sh`
