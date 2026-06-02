# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 11:03] - core_claim
- Implemented and ran SmartPoser's two-stage UWB correction plus pose-estimation RNN on the synthetic held-out-subject benchmark.
- Verified the complete UWB+IMU method improves over the IMU-only ablation on median elbow+wrist error and reduces structured raw UWB distance error.
- `python validate.py --compare` passes with evaluation artifacts written to `scoring/scores.json`.

## Stop Justification
- Completed at milestone `core_claim`; the original public repository is unavailable, so further progress toward paper-scale reproduction would require the released synchronized UWB/IMU/Kinect dataset becoming accessible.
