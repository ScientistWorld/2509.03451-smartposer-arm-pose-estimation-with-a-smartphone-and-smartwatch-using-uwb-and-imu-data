"""Build shortcut submissions that probe the hardened evaluators.

Each shortcut is written in the normal submission layout
``<out>/<name>/fold_<k>/model.pt`` and scored by the unchanged evaluators.
"""

# No `from __future__ import annotations`: the wrappers are compiled with TorchScript.
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

N_FOLDS = 10
WINDOW = 125


class ConstantPose(nn.Module):
    """No model: zero pose, raw range copied through as the 'corrected' range."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, :, :1], torch.zeros_like(x[:, :, :9])


class InflatedRange(nn.Module):
    """Reference pose; range output pushed 10 units above the raw stream."""

    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        return x[:, :, :1] + 10.0, j


class NaNOutputs(nn.Module):
    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        return r * float("nan"), j * float("nan")


class HalfBatch(nn.Module):
    """Subset analogue: answers only the first half of the windows it is given."""

    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        n = x.shape[0] // 2
        return r[:n], j[:n]


class HiddenTable(nn.Module):
    """Reference model plus a 100k-value buffer (capacity hidden outside parameters)."""

    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("table", torch.zeros(100_000))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        return r, j + 0.0 * self.table[0]


class ListTable(nn.Module):
    """Reference model plus a 200k-float list attribute: capacity stored in the
    pickled module state, which neither parameters, buffers nor tensor storage
    count. Caught only by the whole-archive size limit."""

    table: List[float]

    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner
        self.table = [0.0] * 200_000

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        return r, j + 0.0 * self.table[0]


def save(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.eval()
    try:
        scripted = torch.jit.script(module)
    except Exception as exc:
        print(json.dumps({"warning": f"script failed for {path}", "error": repr(exc)}), flush=True)
        scripted = torch.jit.trace(module, torch.zeros(4, WINDOW, 25), check_trace=False)
    scripted.save(str(path))


def copy_submission(src: Path, dst: Path, folds=range(N_FOLDS)) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for k in folds:
        (dst / f"fold_{k}").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f"fold_{k}" / "model.pt", dst / f"fold_{k}" / "model.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--oversized", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Metadata lies are ignored: the evaluator never reads metadata.json.
    copy_submission(args.reference, out / "sc_metadata_zero")
    for k in range(N_FOLDS):
        (out / "sc_metadata_zero" / f"fold_{k}" / "metadata.json").write_text(
            json.dumps({"method": "smartposer", "param_count": 0, "param_count_k": -1000.0}) + "\n"
        )
    # Baseline replacement: a better model under the fixed baseline's name.
    copy_submission(args.reference, out / "imu_only")
    # Incomplete submission: only one fold.
    copy_submission(args.reference, out / "sc_one_fold", folds=[0])
    # Oversized trained model (parameter budget).
    copy_submission(args.oversized, out / "sc_oversized")

    for k in range(N_FOLDS):
        inner_path = args.reference / f"fold_{k}" / "model.pt"
        save(ConstantPose(), out / "sc_constant" / f"fold_{k}" / "model.pt")
        save(InflatedRange(torch.jit.load(str(inner_path))), out / "sc_inflated_range" / f"fold_{k}" / "model.pt")
        save(NaNOutputs(torch.jit.load(str(inner_path))), out / "sc_nan_outputs" / f"fold_{k}" / "model.pt")
        save(HalfBatch(torch.jit.load(str(inner_path))), out / "sc_subset_outputs" / f"fold_{k}" / "model.pt")
        save(HiddenTable(torch.jit.load(str(inner_path))), out / "sc_hidden_table" / f"fold_{k}" / "model.pt")
        save(ListTable(torch.jit.load(str(inner_path))), out / "sc_list_table" / f"fold_{k}" / "model.pt")


if __name__ == "__main__":
    main()
