"""量化整模型 C runtime 单步的访存上限与实测耗时。

p100mw 参数 105.36M, fp32 读取一次 421.5 MB。单步至少要读一遍权重,
因此单步耗时的下界是 421.5MB / 有效带宽。本脚本先测本机可达带宽,
再测 C runtime 单步, 给出两者之比, 判断还有多少空间。
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"


def bandwidth_gbps(nbytes: int, threads: int = 1, iters: int = 5) -> float:
    """用 numpy 分块求和测可达读带宽 (GB/s)。"""
    import os
    if threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    n = nbytes // 4
    a = np.ones(n, dtype=np.float32)
    best = 1e9
    for _ in range(iters):
        t = time.perf_counter()
        s = float(a.sum())
        dt = time.perf_counter() - t
        best = min(best, dt)
    del a
    return nbytes / best / 1e9


def measure_runtime(n_steps: int = 12) -> dict:
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel

    m, cfg, _ = load_model(str(CKPT))
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    taps = d.get("taps", 3)
    h = SnModel(BLOB, d["vocab_size"], d["dim"], d["inner"], d["rank"],
                taps, d["layers"], d.get("group", 8), 1)
    h.reset()
    ts = []
    for i in range(n_steps + 3):
        t = time.perf_counter()
        h.step(100 + i % 64)
        dt = time.perf_counter() - t
        if i >= 3:
            ts.append(dt)
    return {"n": len(ts), "median_ms": statistics.median(ts) * 1e3,
            "min_ms": min(ts) * 1e3, "max_ms": max(ts) * 1e3}


def main():
    print(f"blob: {BLOB}  {BLOB.stat().st_size/1e6:.1f} MB")
    print()
    print("== numpy 可达读带宽 ==")
    bw1 = bandwidth_gbps(400 << 20, threads=1)
    bw6 = bandwidth_gbps(400 << 20, threads=6)
    print(f"  1 线程: {bw1:6.2f} GB/s")
    print(f"  6 线程: {bw6:6.2f} GB/s")
    print()
    print("== C runtime 单步 ==")
    r = measure_runtime()
    print(f"  {r['n']} 步  中位 {r['median_ms']:7.2f} ms  "
          f"min {r['min_ms']:.2f}  max {r['max_ms']:.2f}")
    print()
    size = BLOB.stat().st_size
    print("== 与访存下界比较 ==")
    for label, bw in (("1 线程", bw1), ("6 线程", bw6)):
        floor_ms = size / (bw * 1e9) * 1e3
        print(f"  {label} 带宽下界 {floor_ms:7.2f} ms -> 实测是下界的 "
              f"{r['median_ms']/floor_ms:5.2f}x")
    print(f"  实测有效带宽 {size/(r['median_ms']/1e3)/1e9:6.2f} GB/s")


if __name__ == "__main__":
    sys.exit(main())