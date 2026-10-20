"""分块扫描的块长与 h_min 关系实测 (独占运行).

_safe_chunk 按 p_min 限制块长: max_chunk = headroom_bits*ln2 / (-ln p_min).
h_min=2 -> p_min=0.5 -> 上限 64. 本脚本绕开该限制, 直接对比不同块长的真实耗时,
以判断"放宽 headroom" 或 "抬高 h_min" 是否是速度杠杆.
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


def scan_raw(p, nu, s0, chunk):
    """复刻 logspace_scan 的数学, 不施加 _safe_chunk 限制."""
    g = (1.0 - p) * nu
    T = p.shape[-2]
    lp = torch.log(p)
    s = s0
    outs = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        P = torch.cumsum(lp[..., start:end, :], dim=-2)
        A = torch.cumsum(g[..., start:end, :] * torch.exp(-P), dim=-2)
        s_c = torch.exp(P) * (s.unsqueeze(-2) + A)
        outs.append(s_c)
        s = s_c[..., -1, :]
    return torch.cat(outs, dim=-2), s


def bench(fn, reps=15):
    for _ in range(3):
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
    s0 = torch.zeros(B, N)

    print("=== A. 固定 h_min=2 (p_min=0.5), 变化真实块长 ===")
    p = 0.5 + 0.499 * torch.rand(B, T, N)
    p = p.clamp(0.5, 0.999)
    nu = torch.randn(B, T, N) * 0.1
    print(f"{'chunk':>7s}{'blocks':>8s}{'max_ms':>10s}{'IQR%':>7s}")
    for c in (32, 64, 128, 256, 512, 1024):
        ms, iqr = bench(lambda c=c: scan_raw(p, nu, s0, c))
        print(f"{c:>7d}{math.ceil(T/c):>8d}{ms:>10.2f}{iqr:>7.1f}")

    print("\n=== B. 固定块长 64, 变化 p_min (即 h_min) ===")
    print(f"{'h_min':>7s}{'p_min':>8s}{'ms':>10s}{'IQR%':>7s}")
    for h_min in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        pmin = 1.0 - 1.0 / h_min
        pv = pmin + (1.0 - pmin) * torch.rand(B, T, N)
        ms, iqr = bench(lambda pv=pv: scan_raw(pv, nu, s0, 64))
        print(f"{h_min:>7.1f}{pmin:>8.3f}{ms:>10.2f}{iqr:>7.1f}")

    print("\n=== C. 各 h_min 下的最优块长 (绕开限制全扫) ===")
    print(f"{'h_min':>7s}" + "".join(f"{c:>8d}" for c in (64, 128, 256, 512)))
    for h_min in (2.0, 4.0, 8.0, 16.0):
        pmin = 1.0 - 1.0 / h_min
        pv = (pmin + (1.0 - pmin) * torch.rand(B, T, N)).clamp(max=0.9999)
        row = []
        for c in (64, 128, 256, 512):
            ms, _ = bench(lambda pv=pv, c=c: scan_raw(pv, nu, s0, c))
            row.append(ms)
        best = min(range(4), key=lambda i: row[i])
        cells = "".join(
            f"{row[i]:>7.2f}{'*' if i == best else ' '}" for i in range(4))
        print(f"{h_min:>7.1f}{cells}")


if __name__ == "__main__":
    main()