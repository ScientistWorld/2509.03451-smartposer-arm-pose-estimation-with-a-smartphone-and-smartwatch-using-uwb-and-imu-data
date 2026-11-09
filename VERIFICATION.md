# Verification Report

> Superseded: this report describes the June 2026 prediction-artifact design. The current gym and its audit are in `HARDENING.md`.

## Initial Assessment
The workspace already had a validated `core_claim` reproduction on a synthetic SmartPoser substitute benchmark. The claimed milestone was justified: the method implements the paper's two-stage recurrent UWB/range-plus-inertial pose pipeline, includes an IMU-only comparison, and shows both improved pose error and improved range correction.

## Issues Found
- Minor: active target/experiment docs listed a gross-motor extension that was not present in `scoring/scores.json` and had not been validated by a scheduler run.
- Minor: `scoring/reference.json` included unvalidated extension experiments, which would make the gym rubric less clear than the reproduced baseline.
- Minor: `scripts/run.sh` was still associated with the pending extension in `action.yaml` even though final verification should signal completion of the validated core gym.
- Cosmetic: generated `__pycache__` directories were present locally.

## Fixes Applied
- Removed the unvalidated extension entries from `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, and `scoring/reference.json`.
- Restored active experiment weights to the validated core benchmark proportions: `pose_tracking_synthetic` at 0.7 and `uwb_correction_synthetic` at 0.3.
- Kept `scripts/run.sh` focused on the validated reproducible flow: run `scripts/reproduce.sh`, then `python validate.py --compare`.
- Cleaned generated Python bytecode directories outside `shared/`.
- Ran `python validate.py --compare`; all checks passed.

## Final Judgment
The workspace is ready to serve as a research gym in improve mode and is usable in from-scratch mode with the documented limitation that the original public dataset is unavailable and replaced by a synthetic substitute. Evaluation is separated from `method/`, scored experiments match the active target docs, train/test scoring artifacts are present and independently wired, and scoring emphasizes controllable scientific quantities rather than reproduction fidelity.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: synthetic substitute data only because the paper's cited synchronized UWB/IMU/Kinect release is unavailable.
