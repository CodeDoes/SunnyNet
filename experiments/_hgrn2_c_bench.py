"""HGRN2 C 内核单步基准 (B=1, 单序列).

每格独立子进程, 与既有三架构基准同协议, 便于合并进对照表。
用法: python -X utf8 experiments/_hgrn2_c_bench.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_HGRN2", "F:/sunnyclaw-data/models/hgrn2-1.3B-100B")
BLOB = Path(MDIR) / "hgrn2.c.bin"


def main():
    from sunnynet.runtime.hgrn2 import HGRN2
    from sunnynet.runtime.hgrn2_c import SnHGRN2

    m = HGRN2(MDIR)
    # 参数按结构算: 每层 4*D*D (q/f/i/o) + 3*F*D (mlp) + 4*D, 加 embed/head/norm/lb
    L, D, F = m.n_layers, m.hidden, m._blob.F if m._blob else 5632
    vocab = m.vocab
    n_param = L * (4 * D * D + 3 * F * D + 4 * D) + 2 * vocab * D + D + L * D
    wbytes = BLOB.stat().st_size
    print(f"L={m.n_layers} D={m.hidden} NH={m.n_heads} HK={m.head_f_dim} "
          f"HV={m.head_i_dim} vocab={m.vocab} 参数={n_param/1e6:.1f}M")
    print(f"blob={wbytes/1e6:.1f} MB  理论下限 {wbytes/18.63e9*1e3:.1f} ms "
          f"(顺序读 18.63 GB/s)")

    ids = [1] + list(range(10, 10 + 20))

    # numpy 基线只跑 8 步 (1.3B 的纯 numpy 每步近 300ms, 跑满 20 步不划算);
    # 先跑 2 步预热再计时, 否则首次的页错误与惰性加载会拉高读数
    st = m.reset(m.init_state())
    m.step(ids[0], st)
    m.step(ids[1], st)
    n0 = 8
    t0 = time.perf_counter()
    for t in ids[2:2 + n0]:
        m.step(int(t), st)
    t_np = (time.perf_counter() - t0) / n0 * 1e3

    # C 各线程
    out = {"dims": {"layers": m.n_layers, "hidden": m.hidden,
                    "heads": m.n_heads, "vocab": m.vocab},
           "blob_mb": wbytes / 1e6, "numpy_ms": t_np, "c_ms": {}}
    for th in (1, 4, 8):
        h = SnHGRN2(BLOB, m.vocab, threads=th)
        for t in ids[:3]:
            h.step_raw(int(t))
        h.reset()
        t0 = time.perf_counter()
        for t in ids:
            h.step_raw(int(t))
        dt = (time.perf_counter() - t0) / len(ids) * 1e3
        out["c_ms"][str(th)] = dt
        print(f"C threads={th}: {dt:.2f} ms/tok  ({t_np/dt:.2f}x vs numpy)")

    RES = ROOT / "results"
    RES.mkdir(exist_ok=True)
    (RES / "_hgrn2_c_bench.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote results/_hgrn2_c_bench.json")


if __name__ == "__main__":
    main()