"""缓存悬崖: 有效带宽随工作集大小的变化. 定位 L1/L2/L3 边界。

CPU 原生的核心论点是"把工作集压进缓存"。本脚本给出该论点的直接读数:
同一流式归约, 只改数组大小, 看带宽在哪个尺寸断崖。
"""
from __future__ import annotations

import time

import torch


def bw_bytes(b: int) -> float:
    n = max(1, b // 4)
    x = torch.randn(n)
    reps = max(2, int(1.5e9 / b))
    for _ in range(2):
        x.sum()
    t0 = time.perf_counter()
    for _ in range(reps):
        x.sum()
    dt = (time.perf_counter() - t0) / reps
    return b / dt / 1e9


def main():
    print(f"{'工作集':>10} {'有效带宽':>12}  层级")
    last = None
    for kb in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144):
        b = kb * 1024
        v = bw_bytes(b)
        note = ""
        if last is not None and v < last * 0.6:
            note = "<- 断崖"
        print(f"{kb:>8} KB {v:>10.1f} GB/s  {note}")
        last = v


if __name__ == "__main__":
    main()