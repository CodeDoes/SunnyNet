"""HGRN2 INT4 blob 与 fp32 blob / numpy 参考的数值对照.

INT4 只量化 GEMV 矩阵, 向量与嵌入表保持 fp32, 故差异全部来自权重舍入。
同时列出 blob 体积与逐矩阵量化误差分布。

用法: python -X utf8 experiments/_hgrn2_int4_equiv.py
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


def main():
    from sunnynet.runtime.hgrn2 import HGRN2
    from sunnynet.runtime.hgrn2_c import SnHGRN2

    for p in (BLOB32, BLOB4):
        if not p.is_file():
            raise SystemExit(f"缺 blob: {p}")
    sz32, sz4 = BLOB32.stat().st_size, BLOB4.stat().st_size
    print(f"blob: fp32 {sz32/1e6:.1f} MB -> int4 {sz4/1e6:.1f} MB "
          f"({sz32/sz4:.2f}x)")

    m = HGRN2(MDIR)
    h32 = SnHGRN2(BLOB32, m.vocab)
    h4 = SnHGRN2(BLOB4, m.vocab)
    st = m.reset(m.init_state())

    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))

    # 1) INT4 vs fp32 (同内核, 只差权重精度)
    dmax, worst, am, n = 0.0, -1, 0, 0
    ref32 = None
    for t in ids:
        a = np.asarray(h32.step(int(t)))
        b = np.asarray(h4.step(int(t)))
        d = float(np.abs(a - b).max())
        if d > dmax:
            dmax, worst = d, t
        am += int(a.argmax() == b.argmax())
        n += 1
        ref32 = a
    rel = dmax / max(float(np.abs(ref32).max()), 1e-9)
    print(f"int4 vs fp32 : max|diff|={dmax:.3e} (token={worst})  "
          f"相对={rel:.3e}  argmax {am}/{n}")

    # 2) INT4 vs numpy 参考 (独立实现, 含 fp32 累积误差)
    h4.reset()
    st = m.reset(m.init_state())
    dmax2, am2, n2 = 0.0, 0, 0
    ref = None
    for t in ids:
        ref = m.step(int(t), st)
        b = np.asarray(h4.step(int(t)))
        dmax2 = max(dmax2, float(np.abs(b - ref).max()))
        am2 += int(b.argmax() == ref.argmax())
        n2 += 1
    rel2 = dmax2 / max(float(np.abs(ref).max()), 1e-9)
    print(f"int4 vs numpy: max|diff|={dmax2:.3e}  相对={rel2:.3e}  "
          f"argmax {am2}/{n2}")

    # 3) 逐矩阵量化误差 (直接看最大被量化矩阵的相对误差)
    import torch
    from sunnynet.runtime import quant as Q
    from safetensors import safe_open
    with safe_open(str(Path(MDIR) / "model.safetensors"), framework="pt") as f:
        for name in ("model.layers.0.attn.q_proj.weight",
                     "model.layers.0.attn.i_proj.weight",
                     "model.layers.0.mlp.gate_proj.weight",
                     "lm_head.weight"):
            w = f.get_tensor(name).to(torch.float32).numpy()
            q, s = Q.quant_i4(w, Q.I4_BLOCK)
            wq = Q.dequant_i4(q, s, w.shape[1], Q.I4_BLOCK)
            e = float(np.abs(wq - w).max())
            r = e / max(float(np.abs(w).max()), 1e-9)
            print(f"  {name}: max|err|={e:.3e} 相对={r:.3e}")


if __name__ == "__main__":
    main()