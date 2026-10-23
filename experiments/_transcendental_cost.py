"""CPU 上超越函数 (sigmoid/exp/tanh) 与有理式替代的实测成本.

CPU SIMD 无硬件 sigmoid/exp (AVX2 无相应指令, 只能多项式近似或标量查表),
GPU 有 SFU 专用单元. 这是"架构为谁而生"的一个直接可测维度:
若某架构热路径塞满超越函数, 它在 CPU 上就吃亏。

对照: torch.sigmoid (libm 向量化) vs 自研 poly_sigmoid (仅加/绝对值/除法)。
"""
from __future__ import annotations

import time

import torch


def bench(fn, reps=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def poly_sigmoid(z):
    return 0.5 + 0.5 * (z / (1.0 + z.abs()))


def main():
    print(f"{'宽度':>8} {'sigmoid ms':>12} {'poly ms':>12} {'exp ms':>12} {'tanh ms':>12} {'全宽sigmoid占步骤%':>18}")
    for d in (256, 640, 1280, 2048):
        z = torch.randn(8, 256, d)
        ts = bench(lambda: torch.sigmoid(z))
        tp = bench(lambda: poly_sigmoid(z))
        te = bench(lambda: torch.exp(z))
        tt = bench(lambda: torch.tanh(z))
        # 参照: 一次小模型训练步约 200ms (T=256/B=8/dim=80), 此处按比例给量级
        print(f"{d:>8} {ts*1e3:>12.4f} {tp*1e3:>12.4f} {te*1e3:>12.4f} {tt*1e3:>12.4f} "
              f"{ts/tp:>17.2f}x(poly更快倍数)")
    print("\npoly_sigmoid 相对 sigmoid 的加速倍数 (越高说明超越函数越贵)")


if __name__ == "__main__":
    main()