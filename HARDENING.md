# Hardening report — SmartPoser gym (final iteration)

Hardening took two passes over iteration 11 and this one. The first pass (`notes/hardening/job.sh`, results in `notes/hardening/*.json|md`) rebuilt the gym. The second pass (`notes/hardening/job_v3.sh`, results in `notes/hardening/v3/`) fixed the gaps that remained and checked everything through the final evaluators. Every number below comes from those committed files. Ranks follow the rule in `scoring/reference.json`: weight × |coefficient| × improvement over the stored reference, taken in each metric's better direction. "Normalized" divides each metric's improvement by its standard deviation across the valid variants, as a stand-in for the spread across future attempts (`notes/hardening/rank.py`).

## 1. Do better results now score higher?

**Yes.** On the hidden slice, final evaluator (`notes/hardening/v3/ranks.md`, `paired_noise.md`):

| variant (same method, same data) | median e/w cm | range error removed % | rank raw | rank normalized | paired SD (norm.) |
|---|---:|---:|---:|---:|---:|
| 2-seed ensemble (seeds 18+19, 89.4k params, within the gate) | 7.635 | 69.79 | **+0.941** | **+0.598** | 0.060 |
| random window offsets each epoch (augmentation) | 7.603 | 68.50 | **+0.563** | **+0.522** | 0.061 |
| **reference** (seed 17, 800 epochs) | 7.789 | 68.28 | 0 | 0 | — |
| same-seed rerun of the reference | 7.789 | 68.28 | +0.000 | +0.000 | 0.000 |
| seed 19 | 7.800 | 68.21 | −0.058 | −0.044 | 0.040 |
| seed 18 | 7.884 | 68.28 | −0.208 | −0.222 | 0.052 |
| 600 epochs | 7.883 | 67.87 | −0.296 | −0.222 | 0.047 |
| 1600 epochs (2× training) | 7.934 | 68.94 | −0.337 | −0.484 | 0.104 |
| wider pose network (48 units, 89.7k) | 7.907 | 68.30 | −0.436 | −0.449 | 0.059 |
| 400 / 200 / 100 / 50 / 25 epochs | 8.03 / 8.34 / 9.00 / 10.12 / 10.76 | 67.4 … 47.2 | −0.75 … −13.82 | −0.59 … −9.06 | 0.10–0.28 |
| 1 epoch (fails the pose gate) | floor 22.919 | floor 0 | −62.875 | −50.437 | — |

- **Columns.** Ranks are from `ranks.md`. The paired SD and the CIs below are from `paired_noise.md`, whose normalization also includes the leave-one-out probe, so its point ranks differ by about 0.01.
- **Noise.** The rank spread across three training seeds is 0.208 raw (0.222 normalized). The paired participant bootstrap (48 participants, 300 replicates, every method on the same windows) gives a rank-difference SD of 0.04–0.06 normalized near the reference. Training is deterministic: a same-seed rerun from the final code reproduces the stored reference exactly.
- **Better variants clear the noise.** Both beat the reference by more than twice the seed range, with 95% bootstrap CIs of [+0.51, +0.75] (ensemble) and [+0.43, +0.66] (random offsets). Shorter training scores monotonically worse. Longer, wider or reseeded runs land at or slightly below the reference, because the reference's training length was chosen on the visible slice.
- **Before** (`REDTEAM.md`, old gym): longer training scored +5.591 and one epoch −95.603, but the rank's seed spread was 0.983. Shortcuts outranked every real improvement by one to two orders of magnitude, so ordering by quality did not survive in practice.
- **The paper's effect is far above seed noise.** The IMU-only baseline minus the reference, in median elbow/wrist error, is **1.478 cm** (paired CI [1.324, 1.619]). The seed spread of the reference's median is 0.095 cm (7.789–7.884).

## 2. Major shortcuts: fix, and score before → after

"Before" is the REDTEAM raw rank against the old gym's reference. "After" is the final hidden-slice rank, raw / normalized. A failed run is floored at −62.875 / −50.437 (`notes/hardening/v3/shortcuts_scores_test.json`).

| REDTEAM shortcut | before | fix | after |
|---|---:|---|---:|
| copy joints/range from visible fold test files | +47.05 | Test participants are generated inside `eval/test/` at test time from a private seed; no test inputs or labels exist on disk. Submissions are models (TorchScript, one per fold) that the evaluator runs, not prediction files. | Not expressible. The closest analogue, a lookup table hidden in the model (`sc_hidden_table`: 100k-value buffer; `sc_list_table`: 200k-float pickled list), fails the size and archive gates: **−62.875** |
| report zero parameters | +6.26 | The parameter+buffer count, tensor bytes and archive bytes are measured from the file. Metadata is never read. | `sc_metadata_zero` = reference + forged metadata: **+0.000** (no gain) |
| zero pose + negative parameters | +87.93 | As above, plus a pose gate against the best constant pose | `sc_constant`: failed, **−62.875** |
| inflate the raw comparator by 10 | +460.65 | The raw range stream comes from the evaluator's own inputs | Not expressible. `sc_inflated_range` (range output = raw + 10) fails the range gate: **−62.875** |
| zero pose + inflated raw comparator | +410.13 | Same | `sc_constant`: failed, **−62.875** |
| replace the ground truth with zeros | +67.00 | Ground truth exists only inside the evaluator | Not expressible |
| all-zero arm-span scaling | +41.89 | Arm spans exist only inside the evaluator | Not expressible |
| submit 32 easy windows per fold | +20.99 | The evaluator builds every window and needs a full-shape output for each, from all 10 folds | `sc_subset_outputs` (answers half a batch): failed, **−62.875**. `sc_one_fold`: failed, **−62.875** |
| replace the IMU-only baseline's predictions | +36.32 | `imu_only` is a frozen model inside `eval/*/fixed_models/`, and the reserved names are skipped | `checkpoints/imu_only` is skipped; the fixed row is unchanged (`fixed_baseline_unchanged_under_replacement: true`) |
| forged `scores_test.json` | 0.000 | The evaluator writes the scores | 0.000 (unchanged) |
| NaN outputs | no rank (non-finite metrics) | Finite-output gate, scored at the floor | `sc_nan_outputs`: **−62.875** (failed, with a rank) |
| empty outputs / null metadata | evaluator error | Missing outputs fail at the floor; metadata is ignored | `sc_one_fold`, `sc_bad_layout`: **−62.875** |

New this iteration (gaps left by the first pass):

| probe | before this iteration | fix | after |
|---|---|---|---:|
| `sc_bad_layout` (a submission without the `fold_<k>/` layout) | skipped, no entry, so an empty result cost nothing | Every directory under `checkpoints/` not starting with `_` is a submission. Missing folds fail it. | failed, **−62.875** |
| `sc_tail_garbage` (passes the median gate, throws ~15% of windows five arm spans away) | valid, unclipped raw rank **−550.19**: far *below* a failed run, so failing beat trying | Each scored metric of a valid run is clipped at the floor. A failed run gets the floor on every metric, and `param_count_k` of at least the limit. | **−16.899** / −17.527. Its mean, p90 and high-motion errors (measured 224 / 1427 / 220 cm) are scored at the floor (24.6 / 38.7 / 27.0 cm). |
| `sc_batch_stats` (outputs shifted by a term pooled over the batch) | would have been scored as valid: the evaluator claimed windows could not share information, but a model sees its whole batch | Re-batching probe: 64 windows from the first and last batch, reversed, at a new batch size, must reproduce their outputs within 1e-3 arm spans. Scoring is full fp32 (TF32 off). | failed (gap 3.5e-3), **−62.875** |

The reference, the IMU-only baseline and every honest variant pass the re-batching probe.

## 3. Shortcuts left open, and why

- **Breaking leave-one-out (`sc_all_participants`).** Each fold model is trained on all ten development participants, including the one it should hold out. On the hidden slice this is only 10/9 as much training data, since the hidden participants are still unseen. It scores **+0.180 raw / +0.157 normalized**: below the 0.222 seed range, but consistently positive under paired resampling (CI [+0.10, +0.23]). The hidden test cannot see what a model was trained on, so I found no small fix. On the visible slice the violation is obvious: it scores 6.89 cm there against 7.73 cm on the hidden slice, while honest runs differ by at most 0.15 cm between slices (reference 7.72 / 7.79). A grader can flag it from the train/test gap. `scoring/CONSTRAINTS.md` forbids it.
- **Pooling across the batch below the probe tolerance.** A model whose batch-dependent change stays under 1e-3 arm spans (~0.17 cm) per output would pass the probe. Windows are shuffled across 48 participants and 2-minute recordings, so a batch carries little usable signal. The tolerance sits roughly 1000× above fp32 re-batching noise.
- **Simulator regularities.** The substitute keeps the phone at a fixed point in arm-span units, so the true range is a function of the wrist position. A method can exploit this, and it would not hold for a phone that moves in a real pocket. This is inherent to a synthetic substitute and is not a scoring loophole.

## 4. Reference score before → after

| | old gym (REDTEAM) | after the first pass | **final** |
|---|---:|---:|---:|
| reference median elbow/wrist (cm) | 13.120 | 7.789216 | **7.789206** |
| reference range | MAE 9.004 cm vs raw 12.440 | 68.281% removed (4.2395 cm) | **68.281% (4.2395 cm)** |
| IMU-only median (cm) | 14.192 | 9.267724 | **9.267633** |
| constant pose / raw range | 31.93 cm / 12.44 cm | 22.919 / 13.366 | **22.919 / 13.366** |

The first pass moved the reference to a new simulator (two activity phases, inertial drift; hidden participants generated by the evaluator). It also chose a training length of 800 epochs on the visible leave-one-participant-out slice (`notes/hardening/selection.json`), because the paper gives no epoch count and the old 60-epoch reference was undertrained (REDTEAM: 300 epochs scored +5.59). This iteration changed only the evaluator, so the reference and baseline models were rescored, not retrained. Turning TF32 off moved the scores by at most 1e-4 cm. The same-seed retrain from the final code matches exactly. `scoring/scores_{train,test}.json` and `scoring/scores.json` were regenerated from the reference run alone, with the fixed baselines.

## 5. Headroom check

No scored metric of the reference sits at its bound on the hidden slice. Its pose errors are 7.49 / 8.08 cm median (elbow / wrist), 8.98 cm mean, 16.32 cm p90 and 9.06 cm on high-motion frames, all with a bound of 0. It removes 68.3% of the range error (bound 100%), has an occluded-frame range error of 5.13 cm and a 90th-percentile range error of 8.75 cm (bound 0), and a win rate of 0.75 (bound 1).

Two honest variants score above the reference by well over the noise, and neither comes near a bound: the ensemble (+0.60) and random window offsets (+0.52). So no extra experiment was needed.

The baselines still make sense. IMU-only is 1.48 cm worse than the reference and the constant pose 15.1 cm worse, and neither outranks it. There is no brute-force solution: the task is learned tracking of unseen people, and the test inputs are never available to the method.

The hidden test only runs inference. Scoring one submission (10 fold models × 27,648 windows) takes about 30 s on one GPU.

## 6. Weights and coefficients

Unchanged: pose 0.7, range 0.3. The paper's primary claim is pose accuracy, and range correction is the supporting mechanism it evaluates second. `param_count_k` stays a hard gate with coefficient 0. A scored size term would let a method trade accuracy for size, which the paper never does; the paper fixes a small model. `corrected_uwb_mae_cm` stays at coefficient 0 because it is affine in `uwb_error_reduction_pct`.

## 7. Departures from the paper, and what could not be fixed

- **The paper's data release is unreachable**, so the gym runs on a simulated substitute. Absolute numbers (7.79 cm vs the paper's 11.0 cm, 68% vs 59% range-error reduction) are on the substitute's scale. The paper's numbers stay in `scoring/reference.json` and are not targets. The direction of both claims reproduces: the UWB+IMU model beats IMU-only by 1.48 cm, and the range stream gets more accurate.
- **Scale.** The development data is 10 participants × 12,000 frames (8 min each). The paper had about 28k frames per participant (151k + 129k frames over 10 people). This is a resource scale-down. The protocol is unchanged: leave-one-participant-out, non-overlapping 125-frame training windows, rolling test windows, frame 120 scored.
- **Settings the paper does not specify**, kept as implementation details: batch size 32, gradient-norm clipping at 1.0, and 800 epochs (chosen on the visible slice). The model has 44.7k parameters against the paper's 46.4k (PyTorch LSTM counts; the paper's exact layer details are not given).
- **`validate.py` warnings.** It warns that the evaluators "look detached from the method". This is a false positive: they seed the simulator and the shuffling, and they load models through path variables rather than string literals. All checks pass.

## 8. Files

- **Evaluators:** `eval/train/evaluate.py` (visible) and `eval/test/evaluate.py` + `eval/test/frozen_simulator.py` + `eval/test/fingerprint.json` (hidden). They share no code and do not import `method/`.
- **Fixed baseline:** `eval/{train,test}/fixed_models/imu_only/`.
- **Audit scripts:** `notes/hardening/job_v3.sh`, `make_shortcuts.py`, `make_shortcuts_v3.py`, `make_variants.py`, `train_all_participants.py`, `paired_noise.py`, `rank.py`.
- **Final audit outputs:** `notes/hardening/v3/`.
- **`scripts/run.sh`** reproduces the reference end to end (`scripts/reproduce.sh`, then `validate.py`).
- **`scripts/evaluate_test.sh`** takes no arguments and scores every submission under `checkpoints/`.
