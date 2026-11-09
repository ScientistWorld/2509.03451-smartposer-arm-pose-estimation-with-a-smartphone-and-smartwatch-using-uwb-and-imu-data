"""Shortcut probe: break leave-one-participant-out.

Each fold model is trained by the unchanged reference trainer
(method/train_smartposer.py, same seed, epochs and optimiser) on all ten
development participants, including the one fold k is meant to hold out. On
the visible slice this leaks fold k's own participant; on the hidden slice it
is simply 10/9 as much training data. The probe measures how much each slice
rewards it.

Usage: python notes/hardening/train_all_participants.py --out-dir tmp/h3/shortcuts/sc_all_participants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method"))
import train_smartposer as tsp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    args = argparse.Namespace(
        data_dir=ROOT / "data" / "smartposer_dev",
        out_dir=a.out_dir,
        name=a.out_dir.name,
        epochs=a.epochs,
        snapshot_epochs=None,
        batch_size=32,
        lr=3e-4,
        uwb_loss_weight=1.0,
        pose_hidden=32,
        random_offsets=False,
        seed=a.seed,
        folds=list(range(tsp.N_FOLDS)),
        device=None,
    )
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = tsp.load_subjects(args.data_dir)
    for fold in range(tsp.N_FOLDS):
        target = args.out_dir / f"fold_{fold}"
        if (target / "complete.json").exists() and (target / "model.pt").exists():
            continue
        # train_fold drops index `fold`; a placeholder there keeps all ten real participants.
        padded = subjects[:fold] + [None] + subjects[fold:]
        tsp.train_fold(args, padded, fold, {args.epochs: target}, device)


if __name__ == "__main__":
    main()
