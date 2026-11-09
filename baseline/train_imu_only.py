"""Leave-one-participant-out training of the paper's IMU-only ablation.

Same protocol as the full system (non-overlapping 125-frame windows, Adam lr
3e-4, MPJPE loss) with the range channel removed. One TorchScript model per
fold is written to ``<out-dir>/fold_<k>/model.pt`` (or ``<out-dir>_e<E>/...``
with ``--snapshot-epochs``). The gym's fixed copy of this baseline lives in
``eval/{train,test}/fixed_models/imu_only`` (see scripts/baseline.sh).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from imu_only_model import IMUOnlyPose, mpjpe

WINDOW = 125
N_FOLDS = 10
FEATURE_DIM = 25


def load_subjects(data_dir: Path) -> list[dict[str, np.ndarray]]:
    subjects = []
    for k in range(N_FOLDS):
        path = data_dir / f"subject_{k:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/download.sh")
        with np.load(path) as d:
            subjects.append({"features": d["features"].astype(np.float32), "joints": d["joints"].astype(np.float32)})
    return subjects


def make_windows(subjects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs, js = [], []
    for s in subjects:
        n = s["features"].shape[0] // WINDOW
        xs.append(s["features"][: n * WINDOW].reshape(n, WINDOW, FEATURE_DIM))
        js.append(s["joints"][: n * WINDOW].reshape(n, WINDOW, 9))
    return np.concatenate(xs), np.concatenate(js)


def export(model: torch.nn.Module, fold_dir: Path, meta: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).cpu().eval()
    try:
        scripted = torch.jit.script(cpu_model)
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"warning": "torch.jit.script failed; tracing", "error": repr(exc)}), flush=True)
        scripted = torch.jit.trace(cpu_model, torch.zeros(4, WINDOW, FEATURE_DIM), check_trace=False)
    tmp = fold_dir / "model.pt.tmp"
    scripted.save(str(tmp))
    os.replace(tmp, fold_dir / "model.pt")
    (fold_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (fold_dir / "complete.json").write_text(json.dumps({"epochs": meta["epochs"]}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/smartposer_dev"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/imu_only"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    subjects = load_subjects(args.data_dir)
    for fold in args.folds:
        if args.snapshot_epochs:
            targets = {e: Path(f"{args.out_dir}_e{e}") / f"fold_{fold}" for e in sorted(set(args.snapshot_epochs))}
        else:
            targets = {args.epochs: args.out_dir / f"fold_{fold}"}
        if all((d / "complete.json").exists() and (d / "model.pt").exists() for d in targets.values()):
            continue
        torch.manual_seed(args.seed)
        x, j = make_windows([s for i, s in enumerate(subjects) if i != fold])
        X, J = torch.from_numpy(x).to(device), torch.from_numpy(j).to(device)
        model = IMUOnlyPose().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        started = time.time()
        for epoch in range(1, max(targets) + 1):
            model.train()
            perm = torch.randperm(X.shape[0], device=device)
            total = torch.zeros((), device=device)
            batches = 0
            for start in range(0, X.shape[0], args.batch_size):
                idx = perm[start : start + args.batch_size]
                _, pred = model(X[idx])
                loss = mpjpe(pred, J[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total = total + loss.detach()
                batches += 1
            if epoch in targets:
                meta = {
                    "method": "imu_only",
                    "fold": fold,
                    "held_out_participant": fold,
                    "epochs": epoch,
                    "seed": args.seed,
                    "param_count": sum(p.numel() for p in model.parameters()),
                    "train_loss": float(total) / batches,
                    "training_seconds": time.time() - started,
                }
                export(model, targets[epoch], meta)
                print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
