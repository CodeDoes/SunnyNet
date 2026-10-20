"""验证 _emit 三次 matmul 合并为一次的重参数化等价性.

W_c(y) + b_gain*sn + W_s(fb)
  = [W_c | W_s] @ [y; fb] + b_gain*sn
参数量不变 (W_cat 形状 (dim, inner+dim)), 只是把两次小 matmul 换成一次大 matmul.
本脚本只验证数值等价 + 单算子耗时, 不改模型.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def bench(fn, reps=31, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    B, T, dim, inner = 8, 1024, 512, 896
    y = torch.randn(B, T, inner, requires_grad=True)
    fb = torch.randn(B, T, dim, requires_grad=True)
    sn = torch.randn(B, T, dim)
    Wc = torch.randn(dim, inner) * 0.02
    Ws = torch.randn(dim, dim) * 0.02
    Wcat = torch.cat([Wc, Ws], dim=1)  # (dim, inner+dim)
    b_gain = 0.5

    def split():
        return y @ Wc.t() + b_gain * sn + fb @ Ws.t()

    def merged():
        z = torch.cat([y, fb], dim=-1)
        return z @ Wcat.t() + b_gain * sn

    a, b = split(), merged()
    print(f"数值等价: max|Δ| = {(a - b).abs().max().item():.3e}")

    print(f"三次/两次 matmul: {bench(split):.3f} ms")
    print(f"合并单次 matmul: {bench(merged):.3f} ms")
    print(f"提速: {bench(split) / bench(merged):.2f}x")


if __name__ == "__main__":
    main()