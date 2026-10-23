"""GEMM 效率随行数 M 的曲线: 决定 CPU 原生模型的形状.

固定 N=4096, K=1024 (单个专家的权重块), 扫描行数 M (= 路由到该专家的 token 数).
两个平台各有一条"达到 GFLOPS / 峰值"的饱和曲线, 交点之前的形状对 CPU 更友好.

这直接支撑一个设计判据: 若 MoE 的分组大小落在 CPU 仍高效、GPU 未饱和的区间,
则稀疏 + CPU 是结构上匹配的组合.

用法: python -X utf8 experiments/_gemm_m_curve.py --device cpu|cuda
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


def bench(fn, warmup=5, iters=25):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--ms", default="16,32,64,128,256,512,1024,2048,4096")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)

    N, K = args.N, args.K
    b = torch.randn(K, N, device=dev) * 0.01
    rows = []
    for M in [int(v) for v in args.ms.split(",")]:
        a = torch.randn(M, K, device=dev) * 0.02
        t = bench(lambda: a @ b)
        flops = 2.0 * M * K * N
        rows.append({"M": M, "ms": round(t * 1000, 5),
                     "gflops": round(flops / t / 1e9, 2)})

    print(json.dumps({"device": args.device, "N": N, "K": K, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()