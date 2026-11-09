# Scientist Constraints

## Data
- Train only on the 10 development participants in `data/smartposer_dev/`. The hidden participants are generated inside the evaluator at test time and are never on disk.
- Keep leave-one-participant-out: the fold-`k` model must not see development participant `k` in any form (training, validation, model selection, normalization statistics). Training a fold on its own held-out participant inflates the visible scores and breaks the protocol, even though the hidden slice cannot detect it.
- Do not add external pose, motion-capture or camera data.

## Inputs and outputs
- The model consumes only the provided 25-dimensional sensor window: raw phone-watch range / arm span, phone orientation, watch orientation in the phone frame, and phone/watch accelerations / 30. Ground-truth joints, true distance, arm span and participant identity are not inputs.
- It must return `(range [B,125,1], joints [B,125,9])` in arm-span units. Frame 119 is scored (200 ms look-ahead; later frames in the window are inputs only).
- Each window is processed on its own: outputs must not depend on the other windows in the batch.

## Model scale
- At most 92,800 parameters and buffers per fold model (2x the paper's 46.4k), with at most 742,400 bytes of stored tensors and about 1 MB for the whole TorchScript archive. These are counted by the evaluator from the file. Metadata is ignored.

## Evaluation
- Do not edit `eval/`, the fixed baselines in `eval/*/fixed_models/`, or the reserved submission names `mean_pose`, `raw_uwb`, `imu_only`.

## Compliance Checklist
- [ ] Ten TorchScript fold models under `checkpoints/<method>/fold_<k>/model.pt`.
- [ ] Fold `k` never saw development participant `k`.
- [ ] Inputs limited to the provided sensor window; no batch pooling across windows.
- [ ] Each fold model is within the size gate.
