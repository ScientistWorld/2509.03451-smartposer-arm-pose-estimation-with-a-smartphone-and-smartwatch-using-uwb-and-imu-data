# Train/Test Evaluation Design Notes (current gym)

These notes replace the earlier prediction-artifact design. The final design and its audit are in `HARDENING.md`.

- Submissions are TorchScript models (`checkpoints/<method>/fold_<k>/model.pt`, one per leave-one-participant-out fold). The evaluators build the input windows, run the models, and compute every scored number from the outputs and their own ground truth. Metadata, predictions files and score files written by a submission are never read.
- `eval/train/evaluate.py` (visible): fold-k model on development participant k (`data/smartposer_dev`).
- `eval/test/evaluate.py` (hidden): every fold model on 48 unseen participants generated at test time by `eval/test/frozen_simulator.py` from a private seed. `eval/test/fingerprint.json` guards against numeric drift in that generator.
- The two evaluators share no code, and neither imports `method/`. Fixed baselines (`mean_pose`, `raw_uwb`, frozen `imu_only` models under `eval/*/fixed_models/`) live in `eval/`, and their names are reserved.
- Gates: complete and loadable folds, finite outputs of the right shape, the size limits (params+buffers, tensor bytes, archive bytes), batch independence (re-batched probe), pose better than the constant pose, and range better than the raw stream. A failed run is scored at the floor on every metric. A valid run's metrics are clipped at the floor.
- `validate.py` warns that the evaluators "look detached" because they seed generators and load models via variables rather than literal output paths. This is a false positive: the scores come only from running the submitted models.
