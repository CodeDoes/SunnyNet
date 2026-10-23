"""GLA INT4 blob 与 fp32 的数值与速度对照.

用法: python -X utf8 experiments/_gla_int4_equiv.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_GLA", "E:/sunnyclaw-data/models/gla-340M")
BLOB32 = Path(MDIR) / "gla.c.bin"
BLOB4 = Path(MDIR) / "gla.q4.bin"


def main():
    from sunnynet.runtime.gla_c import SnGLA

    if not BLOB4.is_file():
        raise SystemExit(f"缺 INT4 blob: {BLOB4} (先跑 _export_gla.py --int4)")
    sz32, sz4 = BLOB32.stat().st_size, BLOB4.stat().st_size
    print(f"blob: fp32 {sz32/1e6:.1f} MB -> int4 {sz4/1e6:.1f} MB "
          f"({sz32/sz4:.2f}x)")

    h32 = SnGLA(BLOB32, 32000)
    h4 = SnGLA(BLOB4, 32000)
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))

    h32.reset(); h4.reset()
    rels, am, top5 = [], 0, 0
    for t in ids:
        a = np.asarray(h32.step(int(t)))
        b = np.asarray(h4.step(int(t)))
        rels.append(float(np.abs(a - b).max())
                    / max(float(np.abs(a).max()), 1e-9))
        am += int(a.argmax() == b.argmax())
        top5 += int(int(b.argmax()) in set(np.argsort(a)[-5:].tolist()))
    n = len(ids)
    print(f"int4 vs fp32 : 相对 中位={np.median(rels):.3e} 最大={max(rels):.3e}  "
          f"argmax {am}/{n}  top5 {top5}/{n}")

    # 单步 (state=0) 隔离累积
    h32.reset(); h4.reset()
    a = np.asarray(h32.step(1))
    b = np.asarray(h4.step(1))
    r = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
    print(f"单步 (state=0): 相对={r:.3e}  argmax "
          f"{'一致' if a.argmax() == b.argmax() else '不同'}")

    # 速度
    print()
    for name, p in (("fp32", BLOB32), ("int4", BLOB4)):
        for th in (1, 8):
            h = SnGLA(p, 32000, threads=th)
            for t in ids[:3]:
                h.step_raw(int(t))
            h.reset()
            t0 = time.perf_counter()
            for t in ids:
                h.step_raw(int(t))
            dt = (time.perf_counter() - t0) / len(ids) * 1e3
            print(f"{name} threads={th}: {dt:.2f} ms/tok")


if __name__ == "__main__":
    main()