"""HGRN2 INT4 的误差来源定位: 单步 vs 累积, 以及逐层放大.

逐层对表用 C 侧 probe (stage=9 = o_proj 输出), 看误差从哪一层开始失控。
用法: python -X utf8 experiments/_hgrn2_int4_probe.py
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
BLOB32 = Path(MDIR) / "hgrn2.c.bin"
BLOB4 = Path(MDIR) / "hgrn2.q4.bin"

V = None


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 1.0


def main():
    global V
    import ctypes
    from sunnynet.runtime.hgrn2 import HGRN2
    from sunnynet.runtime.hgrn2_c import SnHGRN2, _load
    V = ctypes.c_void_p

    m = HGRN2(MDIR)
    h32 = SnHGRN2(BLOB32, m.vocab)
    h4 = SnHGRN2(BLOB4, m.vocab)

    # 1) 单步误差 (状态全零, 无累积)
    a = np.asarray(h32.step(1))
    b = np.asarray(h4.step(1))
    rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
    print(f"单步 (state=0) : rel={rel:.4e} cos={_cos(a, b):.6f} "
          f"argmax {'一致' if a.argmax() == b.argmax() else '不同'}")

    # 2) 累积误差增长
    h32.reset(); h4.reset()
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))
    print("\n累积 (逐 token, 状态带误差):")
    for i, t in enumerate(ids):
        a = np.asarray(h32.step(int(t)))
        b = np.asarray(h4.step(int(t)))
        rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
        hit = a.argmax() == b.argmax()
        if i < 6 or i % 6 == 0:
            print(f"  step {i:3d} tok={t:5d}: rel={rel:.4e} "
                  f"cos={_cos(a, b):.6f} argmax={'Y' if hit else 'n'}")

    # 3) 逐层放大 (probe stage=9, 单步)
    print("\n逐层 (stage=9 o_proj 输出, 单步):")
    lib = _load()
    n = m.hidden
    buf32 = (ctypes.c_float * n)()
    buf4 = (ctypes.c_float * n)()
    lib.sn_hg_probe.restype = None
    lib.sn_hg_probe.argtypes = [V, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, V]
    h32.reset(); h4.reset()
    h32.step_raw(1); h4.step_raw(1)
    for l in range(m.n_layers):
        lib.sn_hg_probe(h32._h, 1, l, 9, ctypes.byref(buf32))
        lib.sn_hg_probe(h4._h, 1, l, 9, ctypes.byref(buf4))
        a = np.frombuffer(buf32, dtype=np.float32).copy()
        b = np.frombuffer(buf4, dtype=np.float32).copy()
        rel = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
        if l % 4 == 0 or l == m.n_layers - 1:
            print(f"  layer {l:2d}: rel={rel:.4e} cos={_cos(a, b):.6f}")


if __name__ == "__main__":
    main()