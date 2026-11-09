"""Additional shortcut probes for the v3 evaluators (final hardening iteration).

Written in the normal submission layout next to the v2 probes and scored by the
unchanged evaluators:

* ``sc_batch_stats``  - reference model whose joints are shifted by a term pooled
  over the whole batch (transductive use of the other windows in a batch).
* ``sc_tail_garbage`` - reference model that throws ~15% of windows (picked by an
  input hash) five arm spans away: passes the median gate, but its mean and
  p90 errors are far worse than the constant pose. Probes "a failed run holds
  the worst value of every metric" (floor clipping).
* ``sc_bad_layout``   - a submission directory without the fold layout (a single
  model.pt). Previously skipped, i.e. no entry at all; now scored as failed.

Usage: python notes/hardening/make_shortcuts_v3.py --reference checkpoints/smartposer --out tmp/h3/shortcuts
"""

# No `from __future__ import annotations`: the wrappers are compiled with TorchScript.
import argparse
import shutil
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

N_FOLDS = 10


class BatchStats(nn.Module):
    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        pooled = x[:, :, 10:19].mean(dim=0, keepdim=True).mean(dim=2, keepdim=True)
        own = x[:, :, 10:19].mean(dim=2, keepdim=True)
        return r, j + 0.2 * (pooled - own)


class TailGarbage(nn.Module):
    def __init__(self, inner: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r, j = self.inner(x)
        h = torch.frac(torch.abs(x[:, -1, 22:23]) * 997.0)  # per-window pseudo-random in [0, 1)
        bad = (h > 0.85).to(j.dtype).unsqueeze(1)  # [B, 1, 1]
        return r, j + 5.0 * bad


def save(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(module.eval()).save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for k in range(N_FOLDS):
        inner = args.reference / f"fold_{k}" / "model.pt"
        save(BatchStats(torch.jit.load(str(inner))), out / "sc_batch_stats" / f"fold_{k}" / "model.pt")
        save(TailGarbage(torch.jit.load(str(inner))), out / "sc_tail_garbage" / f"fold_{k}" / "model.pt")
    bad = out / "sc_bad_layout"
    if bad.exists():
        shutil.rmtree(bad)
    bad.mkdir(parents=True)
    shutil.copy2(args.reference / "fold_0" / "model.pt", bad / "model.pt")


if __name__ == "__main__":
    main()
