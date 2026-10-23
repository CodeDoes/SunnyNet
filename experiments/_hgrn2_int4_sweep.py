"""HGRN2 各量化变体的质量/体积/速度对照.

对每个 blob 与 fp32 基准比: 逐 token 相对误差、argmax 命中、top-5 命中、
以及 8 线程 ms/tok。--calib 只做质量对照, --speed 只做速度, 缺省两者都做。

用法: python -X utf8 experiments/_hgrn2_int4_sweep.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_HGRN2",
                      str(Path("F:/sunnyclaw-data/models/hgrn2-1.3B-100B")))
BLOB32 = Path(MDIR) / "hgrn2.c.bin"

VARIANTS = [
    ("int4-all", "hgrn2.q4.bin"),
]


def quality(h32, h4, ids):
    rels, am, top5 = [], 0, 0
    h32.reset(); h4.reset()
    for t in ids:
        a = np.asarray(h32.step(int(t)))
        b = np.asarray(h4.step(int(t)))
        rels.append(float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9))
        am += int(a.argmax() == b.argmax())
        ta = set(np.argsort(a)[-5:].tolist())
        top5 += int(int(b.argmax()) in ta)
    return float(np.median(rels)), float(np.max(rels)), am, top5


def speed(path, vocab, n_rep=20):
    from sunnynet.runtime.hgrn2_c import SnHGRN2
    ids = [1] + list(range(10, 10 + n_rep))
    h = SnHGRN2(path, vocab, threads=8)
    for t in ids[:3]:
        h.step_raw(int(t))
    h.reset()
    t0 = time.perf_counter()
    for t in ids:
        h.step_raw(int(t))
    return (time.perf_counter() - t0) / len(ids) * 1e3


def main():
    from sunnynet.runtime.hgrn2 import HGRN2
    from sunnynet.runtime.hgrn2_c import SnHGRN2

    m = HGRN2(MDIR)
    h32 = SnHGRN2(BLOB32, m.vocab)
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))

    sz32 = BLOB32.stat().st_size
    print(f"{'variant':<18} {'MB':>8} {'x32':>6} {'rel_med':>9} {'rel_max':>9} "
          f"{'argmax':>8} {'top5':>7} {'8T ms':>8}")
    print(f"{'fp32':<18} {sz32/1e6:8.1f} {1.0:6.2f} {'-':>9} {'-':>9} "
          f"{len(ids)}/{len(ids):<5} {len(ids)}/{len(ids):<4} -")

    for name, fn in VARIANTS:
        p = Path(MDIR) / fn
        if not p.is_file():
            print(f"{name:<18} 缺 {fn}")
            continue
        h4 = SnHGRN2(p, m.vocab)
        md, mx, am, t5 = quality(h32, h4, ids)
        ms = speed(p, m.vocab)
        print(f"{name:<18} {p.stat().st_size/1e6:8.1f} {sz32/p.stat().st_size:6.2f} "
              f"{md:9.3e} {mx:9.3e} {am:>3}/{len(ids):<4} {t5:>3}/{len(ids):<3} "
              f"{ms:8.2f}")


if __name__ == "__main__":
    main()