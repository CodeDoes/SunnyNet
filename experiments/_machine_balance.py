"""机器平衡 (machine balance) 实测: FLOPs / 字节.

"GPU 原生 vs CPU 原生"的本质就是这一个比值: 每字节数据能换到多少次浮点运算。
它决定最优架构该偏计算密集还是偏访存密集。

CPU 侧: i7-8750H (6C12T, AVX2), torch 2.14.1+cpu, MKL+oneDNN
GPU 侧: GTX 1060 6GB (sm_61), torch 2.7.1+cu118
"""
from __future__ import annotations

import argparse
import time


def bench_torch(dev: str = "cpu"):
    import torch

    is_cuda = dev == "cuda"
    device = torch.device(dev)

    # --- 峰值 fp32 GEMM (compute) ---
    n = 4096 if is_cuda else 1024
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(3):
        c = a @ b
    if is_cuda:
        torch.cuda.synchronize()
    reps = 50 if is_cuda else 20
    t0 = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    if is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    flops = 2.0 * n ** 3
    gflops = flops / dt / 1e9

    # --- 带宽: 大张量逐元素拷贝 ---
    m = 256 * 1024 * 1024 // 4        # 256MB fp32
    x = torch.randn(m, device=device)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    if is_cuda:
        torch.cuda.synchronize()
    reps_b = 20
    t0 = time.perf_counter()
    for _ in range(reps_b):
        y.copy_(x)
    if is_cuda:
        torch.cuda.synchronize()
    dtb = (time.perf_counter() - t0) / reps_b
    # 拷贝读+写 = 2 * 字节
    bytes_moved = 2 * m * 4
    gbps = bytes_moved / dtb / 1e9

    print(f"[{dev}] GEMM n={n}: {gflops:.1f} GFLOPS | 拷贝带宽: {gbps:.1f} GB/s "
          f"| 平衡 {gflops / gbps:.1f} FLOP/字节")
    return gflops, gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    bench_torch(args.device)


if __name__ == "__main__":
    main()