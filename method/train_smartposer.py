from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from smartposer_model import SmartPoser, mpjpe


def load_npz(path: Path) -> TensorDataset:
    data = np.load(path)
    return TensorDataset(
        torch.from_numpy(data["features"]),
        torch.from_numpy(data["true_distance"]),
        torch.from_numpy(data["joints"]),
        torch.from_numpy(data["arm_span"]),
    )


def evaluate(model: SmartPoser, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    outs = {"corrected_distance": [], "pred_joints": [], "true_distance": [], "true_joints": [], "arm_span": [], "features": []}
    with torch.no_grad():
        for features, true_dist, joints, arm_span in loader:
            features = features.to(device)
            corrected, pred = model(features)
            outs["corrected_distance"].append(corrected.cpu().numpy())
            outs["pred_joints"].append(pred.cpu().numpy())
            outs["true_distance"].append(true_dist.numpy())
            outs["true_joints"].append(joints.numpy())
            outs["arm_span"].append(arm_span.numpy())
            outs["features"].append(features.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/smartposer_synthetic"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/smartposer"))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(load_npz(args.data_dir / "train_windows.npz"), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(load_npz(args.data_dir / "test_windows.npz"), batch_size=args.batch_size)
    model = SmartPoser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, true_dist, joints, _arm_span in train_loader:
            features, true_dist, joints = features.to(device), true_dist.to(device), joints.to(device)
            corrected, pred = model(features)
            loss = torch.mean((corrected - true_dist) ** 2) + mpjpe(pred, joints)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        print(json.dumps(history[-1]), flush=True)

    torch.save({"model": model.state_dict(), "history": history}, args.out_dir / "smartposer.pt")
    preds = evaluate(model, test_loader, device)
    np.savez_compressed(args.out_dir / "predictions.npz", **preds)


if __name__ == "__main__":
    main()
