from __future__ import annotations

import torch
from torch import nn


class UWBCorrector(nn.Module):
    """Paper's UWB correction RNN: 2-layer BiLSTM, 8 hidden units/direction."""

    def __init__(self, input_dim: int = 25) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, 8, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class PoseEstimator(nn.Module):
    """Paper's pose estimator RNN: 2-layer BiLSTM, 32 hidden units/direction."""

    def __init__(self, input_dim: int = 25) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(64, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class SmartPoser(nn.Module):
    """Two-stage SmartPoser: correct raw UWB range, then regress arm joints."""

    def __init__(self) -> None:
        super().__init__()
        self.uwb = UWBCorrector()
        self.pose = PoseEstimator()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corrected = self.uwb(features)
        pose_features = features.clone()
        pose_features[..., :1] = corrected
        joints = self.pose(pose_features)
        return corrected, joints


class IMUOnlyPose(nn.Module):
    """Ablation baseline: same pose estimator with the distance channel zeroed."""

    def __init__(self) -> None:
        super().__init__()
        self.pose = PoseEstimator()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.clone()
        x[..., :1] = 0.0
        return self.pose(x)


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred.reshape(*pred.shape[:-1], 3, 3) - target.reshape(*target.shape[:-1], 3, 3), dim=-1).mean()
