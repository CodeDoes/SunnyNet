"""fp16 路径的线程缩放: 它不再是纯访存受限, 最优线程数可能与 fp32 不同。
每格独立重跑, 逐轮记录。
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
PROMPT = [1] + [100 + (i * 37) % 4000 for i in range(31)]
RAW = BLOB.stat().st_size
# fp16 实际读取字节 (5 个 GEMV 矩阵 + head 减半)
HEAD = 1408 * 8192 * 4
FP16_BYTES = (RAW - HEAD) / 2 + HEAD / 4


def main() -> None:
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads

    _, cfg, _ = load_model(str(CKPT))
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    args = (d["vocab_size"], d["dim"], d["inner"], d["rank"],
            d.get("taps", 3), d["layers"], d.get("group", 8), 1)

    print(f"fp32 blob {RAW/1e6:.1f} MB  fp16 实读约 {FP16_BYTES/1e6:.1f} MB")
    print(f"\n{'线程':>4} {'fp32 中位':>10} {'fp16 中位':>10} {'加速':>7}"
          f" {'fp16 实带宽':>12}")

    for th in (2, 3, 4, 6, 8, 12):
        set_threads(th)
        h32 = SnModel(BLOB, *args)
        h16 = SnModel(BLOB, *args, half=True)
        t32, t16 = [], []
        for r in range(7):
            for tag in (("32", "16") if r % 2 == 0 else ("16", "32")):
                h = h32 if tag == "32" else h16
                h.reset()
                t0 = time.perf_counter()
                for t in PROMPT:
                    h.step(t)
                dt = (time.perf_counter() - t0) / len(PROMPT)
                (t32 if tag == "32" else t16).append(dt)
        m32 = statistics.median(t32) * 1e3
        m16 = statistics.median(t16) * 1e3
        print(f"{th:>4} {m32:>10.2f} {m16:>10.2f} {m32/m16:>6.2f}x"
              f" {FP16_BYTES/(m16/1e3)/1e9:>11.2f}")
        del h32, h16


if __name__ == "__main__":
    main()