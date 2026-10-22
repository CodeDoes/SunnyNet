"""扫描 C runtime 的线程数, 给出单步耗时与有效带宽。

p100mw 单步要读 467.6 MB 权重, 属纯访存受限。线程越多不一定越快:
超过内存控制器饱和点后, 线程调度与带宽争用会反噬。
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
PROMPT = [1] + [100 + (i * 37) % 4000 for i in range(31)]
SIZE = BLOB.stat().st_size


def main():
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads, get_threads

    m, cfg, _ = load_model(str(CKPT))
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    h = SnModel(BLOB, d["vocab_size"], d["dim"], d["inner"], d["rank"],
                d.get("taps", 3), d["layers"], d.get("group", 8), 1)

    print(f"blob {SIZE/1e6:.1f} MB   torch 线程 {torch.get_num_threads()}")
    print()
    print(" 线程   中位 ms   min ms   有效带宽 GB/s   5轮全胜")
    results = {}
    for th in (1, 2, 3, 4, 6, 8, 12):
        set_threads(th)
        ts = []
        for r in range(5):
            h.reset()
            t0 = time.perf_counter()
            for t in PROMPT:
                h.step(t)
            ts.append((time.perf_counter() - t0) / len(PROMPT))
        md = statistics.median(ts) * 1e3
        results[th] = md
        print(f"{th:5d}  {md:9.2f}  {min(ts)*1e3:7.2f}   "
              f"{SIZE/(md/1e3)/1e9:12.2f}   {len(ts)}/5")

    best = min(results, key=results.get)
    base = results[1]
    print()
    print(f"最优线程 {best}  {results[best]:.2f} ms  "
          f"对单线程 {base/results[best]:.2f}x")
    print(f"（set_threads 返回 {set_threads(best)}）")


if __name__ == "__main__":
    main()