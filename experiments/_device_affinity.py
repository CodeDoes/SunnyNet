"""设备亲和性 (device affinity): 同一架构在 CPU 与 GPU 上的相对位置.

问题: 主流架构 (Transformer) 是否为 GPU 而生? CPU 原生架构该长什么样?
做法: 同一参数量对齐的两种架构 (sunnynet / transformer), 同一批输入,
      在 CPU 与 GPU 上各测前向 ms/token, 看谁从 GPU 拿到更多倍数。

读数含义:
- 若某架构的 GPU/CPU 倍数显著更大 -> 它更 "GPU 原生" (从 GPU 获益更多)。
- CPU 原生的目标是让 CPU/GPU 倍数尽量小 (在 CPU 上少吃亏)。

口径: 前向 only, 每配置 warmup 后取中位; 跨架构对照在同一设备内有效。
耗时按设备固定单进程, 不作头条计时 (头条见 train/train.py 独立进程读数)。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from train import build_models  # noqa: E402


def timed_fwd(model, x, reps, sync):
    with torch.no_grad():
        for _ in range(2):
            model(x)
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if sync:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    dev = torch.device(args.device)
    sync = dev.type == "cuda"
    print(f"[{args.device}] preset={args.preset} 前向 ms/token")
    print(f"  {'arch':>12} {'B':>3} {'T':>6} {'ms/token':>12}")
    rows = {}
    for arch in ("sunnynet", "transformer"):
        for B, T in ((1, 256), (1, 2048), (8, 256), (8, 1024)):
            sunny, base, cfg = build_models(arch, args.vocab, args.preset, None, T, 1234)
            model = (sunny if arch == "sunnynet" else base).to(dev).eval()
            x = torch.randint(0, args.vocab, (B, T), device=dev)
            reps = 20 if B == 8 else 30
            dt = timed_fwd(model, x, reps, sync)
            mpt = dt / (B * T) * 1e3
            rows[(arch, B, T)] = mpt
            print(f"  {arch:>12} {B:>3} {T:>6} {mpt:>12.4f}")
    return rows


if __name__ == "__main__":
    main()