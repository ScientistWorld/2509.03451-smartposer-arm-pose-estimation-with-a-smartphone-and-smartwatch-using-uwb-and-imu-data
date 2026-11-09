"""Build composite variant submissions from trained runs (iteration 11 hardening).

``ens``: per fold, the average of two independently seeded SmartPoser models'
range and joint outputs. Two 44.7k-parameter members (89.4k) fit under the
92.8k-parameter gate, so this is an honest, budget-respecting candidate for
"a genuinely better method" that a future agent could also find.

Usage: python notes/hardening/make_variants.py ens <member_a_dir> <member_b_dir> <out_dir>
"""

# No `from __future__ import annotations`: the wrapper is compiled with TorchScript.
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

N_FOLDS = 10


class Ensemble2(nn.Module):
    def __init__(self, a: torch.jit.ScriptModule, b: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ra, ja = self.a(x)
        rb, jb = self.b(x)
        return 0.5 * (ra + rb), 0.5 * (ja + jb)


def main() -> None:
    kind, a_dir, b_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    assert kind == "ens", kind
    for k in range(N_FOLDS):
        dst = out_dir / f"fold_{k}" / "model.pt"
        if dst.exists():
            continue
        a = torch.jit.load(str(a_dir / f"fold_{k}" / "model.pt"), map_location="cpu")
        b = torch.jit.load(str(b_dir / f"fold_{k}" / "model.pt"), map_location="cpu")
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.script(Ensemble2(a, b).eval()).save(str(dst))


if __name__ == "__main__":
    main()
