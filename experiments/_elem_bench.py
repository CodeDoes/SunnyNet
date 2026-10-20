"""元素级算子开销拆分: _compete / _feedback / norm 各自的真实占比.

逐个替换为等价但算子更少的形式, 测耗时差 (数值等价性另行校验).
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def bench(fn, reps=11, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ts.append(time.perf_counter() - t0)
        fn()
    ts = [t for t in ts]
    del ts
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    iqr = (ts[3 * len(ts) // 4] - ts[len(ts) // 4]) / max(med, 1e-12) * 100
    return med * 1000, iqr


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    vocab, B, T = 8192, 8, 1024
    blk = SunnyNet(SunnyConfig.small(vocab)).blocks[0]
    d = blk.dim

    print(f"块维度 dim={d}, 序列 T={T}, batch={B}\n")

    # _compete: roll x2 vs 显式切片
    gate = torch.randn(B, T, d)
    print("=== _compete ===")
    m1, i1 = bench(lambda: blk._compete(gate))
    gg = gate.view(B, T, d // blk.group, blk.group)
    r1 = torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
        torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))
    print(f"  roll 版本        {m1:7.2f} ms (IQR {i1:.1f}%)")

    def compete_slice(gate):
        g = gate.view(B, T, d // blk.group, blk.group)
        a = torch.sigmoid(g - torch.roll(g, -1, dims=-1))
        b = torch.sigmoid(g - torch.roll(g, -2, dims=-1))
        return (a * b).reshape(B, T, d)

    m2, i2 = bench(lambda: compete_slice(gate))
    print(f"  重构版本         {m2:7.2f} ms (IQR {i2:.1f}%)")

    # _feedback: cat + roll x3
    states = torch.randn(B, T, d)
    s_in = torch.randn(B, d)
    print("\n=== _feedback (含 StateCirculant) ===")
    m3, i3 = bench(lambda: blk._feedback(states, s_in))
    print(f"  cat + circ       {m3:7.2f} ms (IQR {i3:.1f}%)")

    def feedback_shift(states, s_in):
        prev = torch.cat([s_in[:, None, :], states[:, :-1]], dim=1)
        a, b, c = blk.circ.t[0], blk.circ.t[1], blk.circ.t[2]
        return a * torch.roll(prev, 1, dims=-1) + b * prev + \
            c * torch.roll(prev, -1, dims=-1)

    m4, i4 = bench(lambda: feedback_shift(states, s_in))
    print(f"  内联版本         {m4:7.2f} ms (IQR {i4:.1f}%)")

    # CSN: mean+var 两次归约
    print("\n=== CenteredScaleNorm ===")
    x = torch.randn(B, T, d)
    m5, i5 = bench(lambda: blk.norm_in(x))
    print(f"  mean+var+rsqrt  {m5:7.2f} ms (IQR {i5:.1f}%)")

    def csn_var_mean(x):
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + blk.norm_in.s_min ** 2 + 1e-5)
        return blk.norm_in.gamma * (x - mu + blk.norm_in.center) * inv

    m6, i6 = bench(lambda: csn_var_mean(x))
    print(f"  显式两步         {m6:7.2f} ms (IQR {i6:.1f}%)")


if __name__ == "__main__":
    main()