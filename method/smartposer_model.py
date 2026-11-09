# No `from __future__ import annotations`: these modules are compiled with TorchScript.
from typing import Tuple

import torch
from torch import nn


class UWBCorrector(nn.Module):
    """Paper's UWB correction RNN: 2-layer BiLSTM, 8 hidden units/direction."""

    def __init__(self, input_dim: int = 25, hidden: int = 8) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class PoseEstimator(nn.Module):
    """Paper's pose estimator RNN: 2-layer BiLSTM, 32 hidden units/direction."""

    def __init__(self, input_dim: int = 25, hidden: int = 32) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


class SmartPoser(nn.Module):
    """Two-stage SmartPoser: correct the raw UWB range, then regress arm joints
    from the corrected range plus the IMU streams. Returns
    (corrected range [B,125,1], joints [B,125,9]) for a 125-frame window."""

    def __init__(self, pose_hidden: int = 32, corrector_hidden: int = 8) -> None:
        super().__init__()
        self.uwb = UWBCorrector(hidden=corrector_hidden)
        self.pose = PoseEstimator(hidden=pose_hidden)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        corrected = self.uwb(features)
        pose_in = torch.cat([corrected, features[:, :, 1:]], dim=2)
        joints = self.pose(pose_in)
        return corrected, joints


def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred.reshape(*pred.shape[:-1], 3, 3) - target.reshape(*target.shape[:-1], 3, 3), dim=-1).mean()
