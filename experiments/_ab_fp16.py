"""fp16 权重路径 vs fp32 的严格 A/B: 数值与速度。

数值: 同一 blob, 同一 token 序列, 逐 token 比 logits 与 argmax。
速度: 同一进程内多轮交错 (fp32/fp16 交替), 逐轮配对比较, 不只看中位。
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

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
PROMPT = [1] + [100 + (i * 37) % 4000 for i in range(31)]
SIZE = BLOB.stat().st_size


def main() -> None:
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads, recommended_threads

    _, cfg, _ = load_model(str(CKPT))
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    args = (d["vocab_size"], d["dim"], d["inner"], d["rank"],
            d.get("taps", 3), d["layers"], d.get("group", 8), 1)

    nth = recommended_threads()
    set_threads(nth)
    print(f"blob {SIZE/1e6:.1f} MB   线程 {nth} (物理核)")

    h32 = SnModel(BLOB, *args)
    h16 = SnModel(BLOB, *args, half=True)

    # 数值
    h32.reset()
    a = np.stack([np.asarray(h32.step(t), dtype=np.float32) for t in PROMPT])
    h16.reset()
    b = np.stack([np.asarray(h16.step(t), dtype=np.float32) for t in PROMPT])
    diff = float(np.abs(a - b).max())
    rel = diff / max(float(np.abs(a).max()), 1e-9)
    am_a, am_b = a.argmax(-1), b.argmax(-1)
    print(f"\n数值 (同 blob, {len(PROMPT)} 步)")
    print(f"  logits max|diff| = {diff:.3e}   相对 {rel:.3e}")
    print(f"  argmax 一致      = {int((am_a == am_b).sum())}/{len(PROMPT)}")

    # 速度: 交错
    reps = 11
    t32, t16 = [], []
    for r in range(reps):
        order = (("32", "16") if r % 2 == 0 else ("16", "32"))
        for tag in order:
            h = h32 if tag == "32" else h16
            h.reset()
            t0 = time.perf_counter()
            for t in PROMPT:
                h.step(t)
            dt = (time.perf_counter() - t0) / len(PROMPT)
            (t32 if tag == "32" else t16).append(dt)

    m32, m16 = statistics.median(t32) * 1e3, statistics.median(t16) * 1e3
    wins = sum(a < b for a, b in zip(t16, t32))
    print(f"\n单步 ms/token ({reps} 轮交错 head-to-head)")
    print(f"  fp32 中位 {m32:7.2f}  min {min(t32)*1e3:7.2f}  "
          f"带宽 {SIZE/(m32/1e3)/1e9:6.2f} GB/s")
    print(f"  fp16 中位 {m16:7.2f}  min {min(t16)*1e3:7.2f}  "
          f"带宽 {SIZE/(m16/1e3)/1e9:6.2f} GB/s (按 fp32 等价字节)")
    print(f"  fp16 胜 {wins}/{reps}   加速 {m32/m16:.2f}x  "
          f"(min 口径 {min(t32)/min(t16):.2f}x)")

    # 每轮配对差
    pair = [(x - y) * 1e3 for x, y in zip(t32, t16)]
    print(f"  逐轮配对差 (fp32-fp16) 中位 {statistics.median(pair):6.2f} ms  "
          f"全为正 {all(p > 0 for p in pair)}")


if __name__ == "__main__":
    main()