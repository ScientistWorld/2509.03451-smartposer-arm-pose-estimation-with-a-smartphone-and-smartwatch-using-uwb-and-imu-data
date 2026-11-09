"""Leave-one-participant-out training of SmartPoser (paper Sec. 3.3-3.5).

For each fold k the model is trained end to end on the other nine development
participants (non-overlapping 125-frame windows, Adam lr 3e-4, loss = UWB MSE +
MPJPE) and exported as TorchScript to ``<out-dir>/fold_<k>/model.pt``. With
``--snapshot-epochs`` the same run is exported at several epochs to
``<out-dir>_e<E>/fold_<k>/model.pt``. Folds that are already complete are
skipped, so an interrupted run can be resumed.
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

from smartposer_model import SmartPoser, mpjpe

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
            subjects.append(
                {
                    "features": d["features"].astype(np.float32),
                    "true_distance": d["true_distance"].astype(np.float32),
                    "joints": d["joints"].astype(np.float32),
                }
            )
    return subjects


def make_windows(subjects: list[dict], offsets: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ds, js = [], [], []
    for s, off in zip(subjects, offsets):
        n = (s["features"].shape[0] - off) // WINDOW
        end = off + n * WINDOW
        xs.append(s["features"][off:end].reshape(n, WINDOW, FEATURE_DIM))
        ds.append(s["true_distance"][off:end].reshape(n, WINDOW, 1))
        js.append(s["joints"][off:end].reshape(n, WINDOW, 9))
    return np.concatenate(xs), np.concatenate(ds), np.concatenate(js)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export(model: torch.nn.Module, fold_dir: Path, meta: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).cpu().eval()
    try:
        scripted = torch.jit.script(cpu_model)
    except Exception as exc:  # pragma: no cover - fallback for TorchScript gaps
        print(json.dumps({"warning": "torch.jit.script failed; tracing", "error": repr(exc)}), flush=True)
        scripted = torch.jit.trace(cpu_model, torch.zeros(4, WINDOW, FEATURE_DIM), check_trace=False)
    tmp = fold_dir / "model.pt.tmp"
    scripted.save(str(tmp))
    os.replace(tmp, fold_dir / "model.pt")
    (fold_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (fold_dir / "complete.json").write_text(json.dumps({"epochs": meta["epochs"]}) + "\n")


def train_fold(args: argparse.Namespace, subjects: list[dict], fold: int, targets: dict[int, Path], device: torch.device) -> None:
    torch.manual_seed(args.seed)
    offset_rng = np.random.default_rng(args.seed * 1000 + fold)
    train_subjects = [s for i, s in enumerate(subjects) if i != fold]

    def to_device(offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, d, j = make_windows(train_subjects, offsets)
        return torch.from_numpy(x).to(device), torch.from_numpy(d).to(device), torch.from_numpy(j).to(device)

    X, D, J = to_device([0] * len(train_subjects))
    model = SmartPoser(pose_hidden=args.pose_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.time()
    max_epoch = max(targets)
    for epoch in range(1, max_epoch + 1):
        if args.random_offsets:
            X, D, J = to_device(offset_rng.integers(0, WINDOW, size=len(train_subjects)).tolist())
        model.train()
        perm = torch.randperm(X.shape[0], device=device)
        total = torch.zeros((), device=device)
        batches = 0
        for start in range(0, X.shape[0], args.batch_size):
            idx = perm[start : start + args.batch_size]
            corrected, pred = model(X[idx])
            loss = args.uwb_loss_weight * torch.mean((corrected - D[idx]) ** 2) + mpjpe(pred, J[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total = total + loss.detach()
            batches += 1
        if epoch in targets:
            meta = {
                "method": args.name,
                "fold": fold,
                "held_out_participant": fold,
                "epochs": epoch,
                "seed": args.seed,
                "pose_hidden": args.pose_hidden,
                "random_offsets": bool(args.random_offsets),
                "uwb_loss_weight": args.uwb_loss_weight,
                "param_count": count_parameters(model),
                "train_loss": float(total) / batches,
                "training_seconds": time.time() - started,
            }
            export(model, targets[epoch], meta)
            print(json.dumps(meta), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/smartposer_dev"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/smartposer"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--uwb-loss-weight", type=float, default=1.0)
    ap.add_argument("--pose-hidden", type=int, default=32)
    ap.add_argument("--random-offsets", action="store_true", help="Variant only: re-draw window offsets every epoch.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.name = args.name or args.out_dir.name

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
        train_fold(args, subjects, fold, targets, device)


if __name__ == "__main__":
    main()
