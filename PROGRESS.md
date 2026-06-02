# Progress

## What Works

- Paper analysis is complete for the core SmartPoser claims and reported metrics.
- `method/` contains a faithful two-stage SmartPoser implementation: UWB correction BiLSTM followed by pose-estimation BiLSTM, trained with the summed UWB MSE and MPJPE losses.
- `data/` contains a method-agnostic synthetic substitute generator because the cited public repository is currently unavailable.
- `eval/` reads prediction artifacts and writes `scoring/scores.json` without importing from `method/`.
- Job `b83e5559-f3d` verified that baseline training, SmartPoser training, artifact writing, evaluation, and validation run end-to-end in the compute container.
- Job `38e42a1e-b7e` reproduced the core claim on the synthetic substitute benchmark: the full UWB+IMU method outperformed the IMU-only ablation and corrected structured UWB range error.

## Results

- Job `b83e5559-f3d` results on the first synthetic split:
  - IMU-only median elbow+wrist error: 19.76 cm.
  - SmartPoser median elbow+wrist error: 16.51 cm.
  - Raw UWB MAE: 3.34 cm.
  - Corrected UWB MAE: 11.24 cm.
- This reaches `method_runs` because the paper's actual two-stage algorithm executes end-to-end and improves pose over the ablation, but it does not reach `core_claim` because UWB correction failed.
- Job `38e42a1e-b7e` results after fixing substitute data difficulty and UWB supervision:
  - IMU-only median elbow+wrist error: 14.18 cm.
  - SmartPoser median elbow+wrist error: 13.39 cm.
  - Raw UWB MAE: 14.09 cm.
  - Corrected UWB MAE: 7.41 cm.
- Paper reference values: IMU-only median elbow+wrist error 14.2 cm, SmartPoser median elbow+wrist error 11.0 cm, raw UWB MAE 21.0 cm, corrected UWB MAE 8.6 cm.

## Remaining

- Higher milestones would require the original released synchronized UWB/IMU/Kinect dataset or additional substitute settings. The core claim environment is usable as packaged.
- If the original repository becomes reachable, replace the synthetic substitute with the released data and rerun.

## Issues

- The paper's cited GitHub repository returned "Repository not found" via `git ls-remote`.
- The first synthetic split made raw UWB too clean on the held-out subject, which let pose improve but did not test the paper's UWB correction claim. The generator now applies more common pose-dependent occlusion and multipath bias.

## Deviations from Paper

- Data/resource substitution: the original synchronized UWB/IMU/Kinect dataset was cited at `https://github.com/FIGLAB/SmartPoser`, but that repository was unreachable on 2026-06-02 (`Repository not found`). I substituted a tracked synthetic generator that preserves the same feature/target structure: raw UWB range with structured occlusion error, phone/watch orientation matrices, accelerations, arm-span-normalized 3-D joints, and held-out-subject evaluation.
- Scale substitution: the final run uses a small synthetic dataset and 60 SmartPoser epochs. The training objective, model topology, window length, and evaluation procedure remain the same.
