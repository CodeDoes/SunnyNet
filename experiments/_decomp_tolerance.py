"""分解容忍度: 同样的总 FLOPs, 拆成 K 个串行小 GEMM, 两设备各损失多少.

这正是稀疏/MoE 的机制: 一个稠密大矩阵乘 = K 个专家各做一小块.
CPU 对小算子分解容忍度高 (调度廉价 + n<=128 本身高效),
GPU 每个 kernel 付启动地板, 拆得越细损失越大.

判据: 分解损失 = 分解后耗时 / 单次大乘耗时. 该值 CPU 应远低于 GPU.

用法: python -X utf8 experiments/_decomp_tolerance.py --device cpu|cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--splits", default="1,2,4,8,16,32")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)

    d, h, N = args.d, args.hidden, args.tokens
    x = torch.randn(N, d, device=dev) * 0.02
    w = torch.randn(d, h, device=dev) * 0.01

    rows = []
    base = None
    for K in [int(v) for v in args.splits.split(",")]:
        hk = h // K
        if hk * K != h:
            continue
        def run():
            outs = []
            for k in range(K):
                outs.append(x @ w[:, k * hk:(k + 1) * hk])
            return outs
        t = bench(run)
        if base is None:
            base = t
        rows.append({"splits": K, "per_gemm_n": hk, "ms": round(t, 4),
                     "loss_vs_1": round(t / base, 3)})

    print(json.dumps({"device": args.device, "tokens": N, "d": d, "hidden": h,
                      "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()