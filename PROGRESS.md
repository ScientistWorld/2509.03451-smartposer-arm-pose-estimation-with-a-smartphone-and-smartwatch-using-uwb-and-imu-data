# Progress

## What Works

- Paper analysis is complete for the core SmartPoser claims and reported metrics.
- `method/` contains a faithful two-stage SmartPoser implementation: UWB correction BiLSTM followed by pose-estimation BiLSTM, trained with the summed UWB MSE and MPJPE losses.
- `data/` contains a method-agnostic synthetic substitute generator because the cited public repository is currently unavailable.
- `eval/` reads prediction artifacts and writes `scoring/scores.json` without importing from `method/`.

## Results

- No compute-node result yet. A test job is prepared to train the baseline and SmartPoser on the synthetic substitute split and compare scores against `scoring/reference.json`.

## Remaining

- Submit the prepared compute job.
- Inspect `scoring/scores.json` after the job and iterate on training length/hyperparameters if the core claim is not supported.
- If the original repository becomes reachable, replace the synthetic substitute with the released data and rerun.

## Issues

- The paper's cited GitHub repository returned "Repository not found" via `git ls-remote`.

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
