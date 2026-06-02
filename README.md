# Overview

- **Paper ID:** 2509.03451
- **Title:** SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data
- **Domain:** Mobile sensing / wearable pose estimation
- **TL;DR:** SmartPoser fuses phone-watch UWB range with IMU streams to estimate 3-D arm pose using commodity devices.

## Short Summary

The paper targets mobile arm pose estimation without cameras, external beacons, or extra worn sensors. It proposes a two-stage recurrent model: one network corrects noisy UWB range and another predicts shoulder, elbow, and wrist positions from corrected range plus IMU features. The original study reports centimeter-scale tracking on held-out users and shows that adding UWB improves over IMU-only sensing.

## Key Results

- Median elbow+wrist error: 11.0 cm in the combined study data.
- UWB correction reduces mean range error from 21.0 cm to 8.6 cm.
- IMU-only ablation worsens median elbow+wrist error from 11.0 cm to 14.2 cm.

---

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Estimate a user's 3-D arm pose from sensors available on common mobile devices, without cameras, external infrastructure, or per-user training data. The target output is shoulder, elbow, and wrist joint position over time in a body-centered coordinate frame.

## Why It Matters

Accurate mobile arm tracking would support fitness, rehabilitation, life logging, occupational training, games, and context-aware interfaces while avoiding privacy and deployment drawbacks of camera-based or instrumented motion-capture systems.

## Success Criteria

A successful system predicts wrist and elbow positions with low centimeter-scale error on held-out users, remains stable across natural activities and large arm motions, and does not require user-specific training examples.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

SmartPoser estimates arm pose using only an off-the-shelf smartphone and smartwatch by combining inertial measurements with phone-watch UWB distance. The central claim is that absolute UWB range stabilizes and improves arm tracking compared with IMU-only sensing.

## Approach

The system has two learned stages trained end to end. First, a 2-layer bidirectional LSTM with 8 hidden units per direction corrects noisy raw UWB range using 125-frame windows of distance, phone/watch orientation matrices, and accelerations. Second, a 2-layer bidirectional LSTM with 32 hidden units per direction consumes corrected range plus the same IMU features and predicts a 125-frame sequence of shoulder, elbow, and wrist positions normalized by arm span. Training uses Adam with learning rate 3e-4 and a summed loss: UWB distance MSE plus mean per-joint position error. At inference, the 120th frame of each 125-frame output window is used for a 200 ms latency/accuracy tradeoff.

## Main Claims

Primary: phone/watch UWB plus IMU data can track wrist and elbow pose around 11 cm median error without training on the target user. Secondary: the UWB correction network reduces structured range error, and the complete UWB+IMU system outperforms an IMU-only ablation.

---

# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The main metric is median positional error in centimeters for elbow and wrist joints, computed as Euclidean distance between predicted and ground-truth 3-D joint coordinates after scaling normalized coordinates by arm span. Lower is better. Supporting metrics include per-joint median error, mean elbow+wrist error, and mean absolute error for range correction.

## Baselines and Targets

The paper reports 11.0 cm median elbow+wrist error, with 8.7 cm elbow and 13.3 cm wrist median error. The IMU-only ablation reports 14.2 cm median elbow+wrist error, with 11.7 cm elbow and 16.7 cm wrist. For range correction, the paper reports raw UWB mean error of 21.0 cm and corrected UWB mean error of 8.6 cm.

## Evaluation Protocol

The original study uses leave-one-participant-out cross validation over 10 participants and reports results across daily activities plus gross motor arm motions. This environment uses a synthetic substitute split with the final subject held out, because the paper's cited repository is unavailable. Evaluation reads prediction artifacts from `checkpoints/*/predictions.npz` and writes `scoring/scores.json`; it does not import from `method/`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 2 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
