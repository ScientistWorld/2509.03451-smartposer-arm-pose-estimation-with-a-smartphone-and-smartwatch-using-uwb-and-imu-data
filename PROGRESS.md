# Progress

## What Works

- Paper analysis is complete for the core SmartPoser claims and reported metrics.
- `method/` contains a faithful two-stage SmartPoser implementation: UWB correction BiLSTM followed by pose-estimation BiLSTM, trained with the summed UWB MSE and MPJPE losses.
- `data/` contains a method-agnostic synthetic substitute generator because the cited public repository is currently unavailable.
- `eval/` reads prediction artifacts and writes `scoring/scores.json` without importing from `method/`.
- Job `b83e5559-f3d` verified that baseline training, SmartPoser training, artifact writing, evaluation, and validation run end-to-end in the compute container.

## Results

- Job `b83e5559-f3d` results on the first synthetic split:
  - IMU-only median elbow+wrist error: 19.76 cm.
  - SmartPoser median elbow+wrist error: 16.51 cm.
  - Raw UWB MAE: 3.34 cm.
  - Corrected UWB MAE: 11.24 cm.
- This reaches `method_runs` because the paper's actual two-stage algorithm executes end-to-end and improves pose over the ablation, but it does not reach `core_claim` because UWB correction failed.

## Remaining

- Submit the updated compute job with harder structured UWB errors, longer training, and stronger UWB supervision.
- Inspect `scoring/scores.json` after the job and iterate further if UWB correction or pose improvement still fails.
- If the original repository becomes reachable, replace the synthetic substitute with the released data and rerun.

## Issues

- The paper's cited GitHub repository returned "Repository not found" via `git ls-remote`.
- The first synthetic split made raw UWB too clean on the held-out subject, which let pose improve but did not test the paper's UWB correction claim. The generator now applies more common pose-dependent occlusion and multipath bias.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

- Data/resource substitution: the original synchronized UWB/IMU/Kinect dataset was cited at `https://github.com/FIGLAB/SmartPoser`, but that repository was unreachable on 2026-06-02 (`Repository not found`). I substituted a tracked synthetic generator that preserves the same feature/target structure: raw UWB range with structured occlusion error, phone/watch orientation matrices, accelerations, arm-span-normalized 3-D joints, and held-out-subject evaluation.
- Scale substitution: the prepared job uses fewer epochs than a full reproduction for the first smoke run. The training objective, model topology, window length, and evaluation procedure remain the same.
