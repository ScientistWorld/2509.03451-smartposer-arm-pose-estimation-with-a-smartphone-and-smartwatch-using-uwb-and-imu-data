# No `from __future__ import annotations`: these modules are compiled with TorchScript.
from typing import Tuple

import torch
from torch import nn


class PoseEstimator(nn.Module):
    """Pose estimator with the paper's capacity: 2-layer BiLSTM, 32 units/direction."""

    def __init__(self, input_dim: int = 25, hidden: int = 32) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class IMUOnlyPose(nn.Module):
    """Paper's IMU-only ablation: the pose estimator with the range channel zeroed.

    It performs no range correction, so its range output is the raw stream."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.pose = PoseEstimator(hidden=hidden)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([torch.zeros_like(features[:, :, :1]), features[:, :, 1:]], dim=2)
        return features[:, :, :1], self.pose(x)


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred.reshape(*pred.shape[:-1], 3, 3) - target.reshape(*target.shape[:-1], 3, 3), dim=-1).mean()
