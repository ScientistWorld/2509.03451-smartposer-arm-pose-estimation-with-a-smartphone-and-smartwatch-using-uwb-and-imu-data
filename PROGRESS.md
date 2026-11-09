# Progress

## Current state (final hardening, 2026-11-09)

The gym's current design, results and audit are in `HARDENING.md`, `briefing/evaluation.md` and `scoring/`. Earlier entries below describe superseded designs: prediction-file artifacts and the `data/synthetic_smartposer.py` generator.

## What Works

- Paper analysis is complete for the core SmartPoser claims and reported metrics.
- An independent paper-read subagent confirmed the reproduction-critical algorithm details: two-stage UWB-correction plus pose-estimation BiLSTMs, 125-frame windows, n-5 output scoring, end-to-end summed UWB MSE + MPJPE training, subject-disjoint evaluation, IMU-only ablation, and raw-vs-corrected UWB metrics.
- `method/` contains a faithful two-stage SmartPoser implementation: UWB correction BiLSTM followed by pose-estimation BiLSTM, trained with the summed UWB MSE and MPJPE losses.
- `data/` contains a method-agnostic synthetic substitute generator because the cited public repository is currently unavailable.
- `eval/` reads prediction artifacts and writes `scoring/scores.json` without importing from `method/`.
- Job `b83e5559-f3d` verified that baseline training, SmartPoser training, artifact writing, evaluation, and validation run end-to-end in the compute container.
- Job `38e42a1e-b7e` reproduced the core claim on the first synthetic substitute benchmark: the full UWB+IMU method outperformed the IMU-only ablation and corrected structured UWB range error.
- Continuation audit on 2026-06-02 found integrity gaps in that first `core_claim`: the synthetic protocol used only one held-out subject, the run used an 8x UWB loss weight instead of the paper's unweighted summed loss, parameter counts were hard-coded in evaluation, and evaluation did not fail on missing required artifacts.
- Fixes are now implemented: fold-selectable synthetic data, 10-fold leave-one-synthetic-subject-out run scripts, paper-default `UWB MSE + MPJPE` training, metadata-recorded trainable parameter counts, and stricter evaluation artifact requirements. A new job is required before restoring `core_claim`.
- Job `1ee26c07-551` ran the corrected 10-fold substitute benchmark successfully and passed `python validate.py --compare`. The corrected run restores `core_claim` on the packaged substitute benchmark.
- The IMU-only baseline has been decoupled from `method/` so baseline training no longer imports the proposed method module.
- Job `45fc335a-49e` ran the corrected 10-fold substitute benchmark after the continuation audit fixes: rolling one-frame test windows, 5 Hz raw UWB repeated to 25 Hz, and phone-relative watch orientation/acceleration. The job passed `python validate.py --compare` and restores `core_claim` for the current revision.
- On 2026-06-02 12:52, the default evaluator was hardened to emit the robustness/constraint metrics already used by the train/test evaluators: p90 pose error, high-motion pose error, forearm-length error, corrected UWB p90, hard-UWB improvement, and UWB correction win rate.
- On 2026-06-02 13:02, `scripts/run.sh` was fixed to regenerate synthetic data and prediction artifacts from source instead of relying on gitignored checkpoints. Final verification removed an unvalidated gross-motor extension from active scoring docs/reference so the packaged gym advertises only validated `core_claim` results.

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
- Job `1ee26c07-551` corrected 10-fold substitute results with paper-default loss:
  - IMU-only median elbow+wrist error: 14.76 cm.
  - SmartPoser median elbow+wrist error: 13.55 cm.
  - Raw UWB MAE: 14.01 cm.
  - Corrected UWB MAE: 9.41 cm.
  - Trainable parameters from metadata: IMU-only 40.777k, SmartPoser 44.698k.
- Job `45fc335a-49e` corrected rolling-window 10-fold substitute results:
  - IMU-only median elbow+wrist error: 14.73 cm.
  - SmartPoser median elbow+wrist error: 13.66 cm.
  - Raw UWB MAE: 14.02 cm.
  - Corrected UWB MAE: 9.48 cm.
  - Trainable parameters from metadata: IMU-only 40.777k, SmartPoser 44.698k.
- Paper reference values: IMU-only median elbow+wrist error 14.2 cm, SmartPoser median elbow+wrist error 11.0 cm, raw UWB MAE 21.0 cm, corrected UWB MAE 8.6 cm.

## Remaining

- Higher milestones would require the original released synchronized UWB/IMU/Kinect dataset or additional substitute settings that are separately run and validated.
- If the original repository becomes reachable, replace the synthetic substitute with the released data and rerun.

## Issues

- The paper's cited GitHub repository returned "Repository not found" via `git ls-remote`.
- The first synthetic split made raw UWB too clean on the held-out subject, which let pose improve but did not test the paper's UWB correction claim. The generator now applies more common pose-dependent occlusion and multipath bias.
- The login-node Python environment does not have `numpy`, so `scripts/evaluate.sh` is expected to run in the packaged container/overlay or scheduler job environment. The completed compute job `38e42a1e-b7e` verified that path. A local 2026-06-02 smoke run of `bash scripts/evaluate.sh` failed with `ModuleNotFoundError: No module named 'numpy'` after Python syntax checks and `python validate.py --compare` passed.
- The first continuation audit downgraded the claimed milestone until the corrected protocol is run.
- The corrected benchmark restores the core effect but does not reach the paper's 11.0 cm real-data primary metric. It is a substitute-data core reproduction, not a full paper-scale reproduction.

## Deviations from Paper

- Data/resource substitution: the original synchronized UWB/IMU/Kinect dataset was cited at `https://github.com/FIGLAB/SmartPoser`, but that repository was unreachable on 2026-06-02 (`Repository not found`). I substituted a tracked synthetic generator that preserves the same feature/target structure: raw UWB range with structured occlusion error, phone/watch orientation matrices, accelerations, arm-span-normalized 3-D joints, and held-out-subject evaluation.
- Scale substitution: the final run uses a small synthetic dataset and 60 SmartPoser epochs. The training objective, model topology, window length, and evaluation procedure remain the same.
