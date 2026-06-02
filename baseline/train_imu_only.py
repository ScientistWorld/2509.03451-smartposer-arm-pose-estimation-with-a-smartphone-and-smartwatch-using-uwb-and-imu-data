from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz(path: Path) -> TensorDataset:
    data = np.load(path)
    return TensorDataset(
        torch.from_numpy(data["features"]),
        torch.from_numpy(data["true_distance"]),
        torch.from_numpy(data["joints"]),
        torch.from_numpy(data["arm_span"]),
    )


class PoseEstimator(torch.nn.Module):
    """IMU-only ablation network matching the paper's pose-estimator capacity."""

    def __init__(self, input_dim: int = 25) -> None:
        super().__init__()
        self.rnn = torch.nn.LSTM(input_dim, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.out = torch.nn.Linear(64, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class IMUOnlyPose(torch.nn.Module):
    """Ablation baseline: same temporal pose estimator with the range channel removed."""

    def __init__(self) -> None:
        super().__init__()
        self.pose = PoseEstimator()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.clone()
        x[..., :1] = 0.0
        return self.pose(x)


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_j = pred.reshape(*pred.shape[:-1], 3, 3)
    target_j = target.reshape(*target.shape[:-1], 3, 3)
    return torch.linalg.norm(pred_j - target_j, dim=-1).mean()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/smartposer_synthetic"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints/imu_only"))
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(load_npz(args.data_dir / "train_windows.npz"), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(load_npz(args.data_dir / "test_windows.npz"), batch_size=args.batch_size)
    model = IMUOnlyPose().to(device)
    param_count = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, _true_dist, joints, _arm_span in train_loader:
            features, joints = features.to(device), joints.to(device)
            pred = model(features)
            loss = mpjpe(pred, joints)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)

    outs = {"pred_joints": [], "true_joints": [], "arm_span": [], "features": [], "true_distance": []}
    model.eval()
    with torch.no_grad():
        for features, true_dist, joints, arm_span in test_loader:
            pred = model(features.to(device)).cpu().numpy()
            outs["pred_joints"].append(pred)
            outs["true_joints"].append(joints.numpy())
            outs["arm_span"].append(arm_span.numpy())
            outs["features"].append(features.numpy())
            outs["true_distance"].append(true_dist.numpy())
    np.savez_compressed(args.out_dir / "predictions.npz", **{k: np.concatenate(v, axis=0) for k, v in outs.items()})
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "method": "imu_only",
                "param_count": param_count,
                "param_count_k": param_count / 1000.0,
                "epochs": args.epochs,
                "data_dir": str(args.data_dir),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
