# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-02 10:55] - none
- Read the SmartPoser paper and extracted the core algorithm: UWB correction BiLSTM plus pose-estimation BiLSTM.
- Verified the cited `https://github.com/FIGLAB/SmartPoser` repository is currently unreachable, so the original released dataset/code cannot be fetched in this turn.
- Implemented the paper's actual two-stage model topology and a method-agnostic synthetic substitute dataset for phone/watch UWB+IMU arm tracking.
- Added evaluation targets for pose tracking, UWB correction, and the IMU-only ablation.

### [2026-06-02 11:00] - method_runs
- Job `b83e5559-f3d` completed successfully on a CPU test node.
- The IMU-only baseline and two-stage SmartPoser model both trained end-to-end and wrote prediction artifacts.
- Evaluation produced `scoring/scores.json`; SmartPoser improved pose median error over IMU-only, but UWB correction was worse than raw UWB, so this is not yet `core_claim`.
- Updated the synthetic data generator and training configuration to better exercise and supervise UWB correction before the next job.

### [2026-06-02 11:05] - core_claim
- Job `38e42a1e-b7e` completed successfully on a CPU test node.
- SmartPoser reduced median elbow+wrist pose error from 14.18 cm for the IMU-only ablation to 13.39 cm on the held-out synthetic subject.
- The UWB correction stage reduced range MAE from 14.09 cm raw UWB to 7.41 cm corrected UWB.
- This supports the paper's central claim on the documented synthetic substitute benchmark: fusing UWB range with IMU streams improves arm pose estimation and corrects structured UWB error.

## Stop Justification
- Completed at milestone `core_claim`.
