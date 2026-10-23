"""Mamba-2 C 内核与 numpy 参考逐 token 对照.

用法: python -X utf8 experiments/_mamba2_c_equiv.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))
BLOB = Path(MDIR) / "mamba2.c.bin"


def main():
    from sunnynet.runtime.mamba2 import Mamba2
    from sunnynet.runtime.mamba2_c import SnMamba2

    m = Mamba2(MDIR)
    h = SnMamba2(BLOB, m.vocab)
    print(f"vocab={m.vocab} layers={m.n_layers} D={m.d_model} NH={m.n_heads} "
          f"HD={m.head_dim} NS={m.state_size}")

    st = m.reset(m.init_state())
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))
    dmax, am, n = 0.0, 0, 0
    worst = -1
    for t in ids:
        ref = m.step(int(t), st)
        c = np.asarray(h.step(int(t)))
        d = float(np.abs(c - ref).max())
        if d > dmax:
            dmax, worst = d, t
        am += int(c.argmax() == ref.argmax())
        n += 1
    rel = dmax / max(float(np.abs(ref).max()), 1e-9)
    print(f"逐 token: max|diff|={dmax:.3e} (最大处 token={worst})  "
          f"相对={rel:.3e}  argmax {am}/{n}")


if __name__ == "__main__":
    main()