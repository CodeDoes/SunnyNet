"""转置扫描下, 块长与 p_min 的耗时规律 (旧结论"块越大越慢"是否仍成立).

旧结论: chunk=64 最优, 放大块长更慢 (非转置 cumsum 沿 stride=N 维归约,
段越长越慢). 转置后 cumsum 沿末维步长 1, 缩放规律可能反转 ——
若反转, 则抬高 h_min (放宽 _safe_chunk 上限) 可大幅减少块数, 成为速度杠杆.
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.ops import gated_scan  # noqa: E402


def bench(fn, reps=15, warm=4):
    for _ in range(warm):
        fn()
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
    B, T, N = 8, 1024, 512
    g = torch.randn(B, T, N) * 0.1
    s0 = torch.zeros(B, N)

    print("=== 转置扫描: 变化块长 (h_min=2, p_min=0.5, 绕开上限) ===")
    p = (torch.rand(B, T, N) * 0.5 + 0.5).clamp(max=0.999)
    print(f"{'chunk':>7s}{'blocks':>8s}{'ms':>10s}{'IQR%':>7s}")
    best = None
    for c in (16, 32, 64, 128, 256, 512, 1024):
        ms, iqr = bench(lambda c=c: gated_scan(p, g, s0, chunk=c,
                                              p_min_hint=1e-9))
        if best is None or ms < best[1]:
            best = (c, ms)
        print(f"{c:>7d}{math.ceil(T / c):>8d}{ms:>10.2f}{iqr:>7.1f}")
    print(f"  最优块长 = {best[0]} ({best[1]:.2f} ms)")

    print("\n=== 变化 h_min (即真实 p_min), 块长固定 64 ===")
    print(f"{'h_min':>7s}{'p_min':>8s}{'ms':>10s}{'IQR%':>7s}")
    for h_min in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        pmin = 1.0 - 1.0 / h_min
        pv = pmin + (1.0 - pmin) * torch.rand(B, T, N)
        ms, iqr = bench(lambda pv=pv: gated_scan(pv, g, s0, chunk=64,
                                                 p_min_hint=pmin))
        print(f"{h_min:>7.1f}{pmin:>8.3f}{ms:>10.2f}{iqr:>7.1f}")

    print("\n=== 组合: 各 h_min 用其允许的最大块长 ===")
    print(f"{'h_min':>7s}{'cap':>6s}{'blocks':>8s}{'ms':>10s}{'IQR%':>7s}")
    for h_min in (2.0, 4.0, 8.0, 16.0, 32.0):
        pmin = 1.0 - 1.0 / h_min
        cap = max(1, int(64 * math.log(2) / (-math.log(pmin))))
        pv = pmin + (1.0 - pmin) * torch.rand(B, T, N)
        ms, iqr = bench(lambda pv=pv, cap=cap: gated_scan(
            pv, g, s0, chunk=cap, p_min_hint=pmin))
        print(f"{h_min:>7.1f}{cap:>6d}{math.ceil(T / cap):>8d}{ms:>10.2f}{iqr:>7.1f}")


if __name__ == "__main__":
    main()