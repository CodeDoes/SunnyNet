"""HGRN2 C 内核与 numpy 参考逐 token 对照.

用法: python -X utf8 experiments/_hgrn2_c_equiv.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_HGRN2",
                      str(Path("F:/sunnyclaw-data/models/hgrn2-1.3B-100B")))
BLOB = Path(MDIR) / "hgrn2.c.bin"


def main():
    from sunnynet.runtime.hgrn2 import HGRN2
    from sunnynet.runtime.hgrn2_c import SnHGRN2

    m = HGRN2(MDIR)
    h = SnHGRN2(BLOB, m.vocab)
    print(f"vocab={m.vocab} layers={m.n_layers} D={m.hidden} NH={m.n_heads} "
          f"HK={m.head_f_dim} HV={m.head_i_dim} lb={m.use_lower_bound}")

    st = m.reset(m.init_state())
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))
    dmax, am, n = 0.0, 0, 0
    worst = -1
    ref = None
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