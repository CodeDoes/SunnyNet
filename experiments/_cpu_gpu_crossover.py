"""CPU 与 GPU 在"问题规模"上的交叉点实测.

机器平衡相同 (都 ~18.7 FLOP/字节) 说明瓶颈不在此。真正的分歧在:
小规模/小 batch 下 GPU 的并行宽度吃不满, CPU 反而高效。
本脚本扫 GEMM 尺寸与 B=1 单序列解码, 找出两者各自的适用区间。
"""
from __future__ import annotations

import argparse
import time

import torch


def t_of(fn, reps, sync):
    for _ in range(3):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def scan_gemm(dev):
    sync = dev == "cuda"
    print(f"[{dev}] GEMM 尺寸扫描 (GFLOPS)")
    print(f"  {'n':>6} {'ms':>10} {'GFLOPS':>10}")
    for n in (64, 128, 256, 512, 1024, 2048, 4096):
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        reps = 200 if n <= 256 else (100 if n <= 1024 else 30)
        dt = t_of(lambda: a @ b, reps, sync)
        fl = 2.0 * n ** 3 / dt / 1e9
        print(f"  {n:>6} {dt*1e3:>10.4f} {fl:>10.1f}")


def scan_decode(dev):
    """单序列 (B=1) 小矩阵逐 token: 模拟解码态的 GEMV 负载."""
    sync = dev == "cuda"
    print(f"[{dev}] B=1 单序列 GEMV (dim x dim, 每 token 一次)")
    print(f"  {'dim':>6} {'us/token':>12} {'GFLOPS':>10}")
    for d in (256, 512, 1024, 2048, 4096):
        w = torch.randn(d, d, device=dev)
        x = torch.randn(1, d, device=dev)
        reps = 2000 if d <= 1024 else 500
        dt = t_of(lambda: x @ w.t(), reps, sync)
        fl = 2.0 * d * d / dt / 1e9
        print(f"  {d:>6} {dt*1e6:>12.3f} {fl:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    scan_gemm(args.device)
    scan_decode(args.device)


if __name__ == "__main__":
    main()