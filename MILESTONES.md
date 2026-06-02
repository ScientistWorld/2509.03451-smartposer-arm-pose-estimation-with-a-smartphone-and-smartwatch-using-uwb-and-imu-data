# Reproduction Milestones

**Current: none**

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

## Stop Justification
- Not stopped.
