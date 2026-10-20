"""线程数与扫描实现方式的最终判定 (交叉测量消除漂移).

1. 线程数: 按 4,6,8,4,6,8 交叉顺序测量, 抵消负载/热漂移.
2. torch.associative_scan: 是否可替代 Python 分块循环.
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


def bench(fn, reps=9, warm=3):
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
    vocab, B, T = 8192, 8, 1024

    print("=== 线程数交叉测量 (T=1024 B=8, fwd+bwd) ===")
    models = {}
    for th in (4, 6, 8):
        torch.set_num_threads(th)
        torch.manual_seed(0)
        m = SunnyNet(SunnyConfig.small(vocab)).train()
        x = torch.randint(0, vocab, (B, T))
        y = torch.randint(0, vocab, (B, T))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

        def step(m=m, x=x, y=y, opt=opt):
            lg = m(x)
            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        models[th] = step

    acc = {4: [], 6: [], 8: []}
    for rnd in range(3):
        for th in (4, 6, 8):
            torch.set_num_threads(th)
            acc[th].append(bench(models[th], reps=7, warm=2))
    for th in (4, 6, 8):
        v = acc[th]
        print(f"  threads={th}  轮次={[f'{x:.0f}' for x in v]}  "
              f"中位={statistics.median(v):.1f} ms")

    print("\n=== torch.associative_scan 可用性 ===")
    if not hasattr(torch, "associative_scan"):
        print("  本版本 torch 无 associative_scan")
        return
    from sunnynet.ops import gated_scan
    N = 512
    p = (torch.rand(B, T, N) * 0.5 + 0.5).clamp(max=0.999)
    g = torch.randn(B, T, N) * 0.1
    s0 = torch.zeros(B, N)

    def combine(a, b):
        return a[0] * b[0], b[0] * a[1] + b[1]

    try:
        def scan_assoc(p, g):
            return torch.associative_scan((p, g), dim=-2, combine_fn=combine)
        t0 = time.perf_counter()
        out, _ = scan_assoc(p, g)
        print(f"  首次耗时 {time.perf_counter() - t0:.2f} s")
        ref, _ = gated_scan(p, g, s0, chunk=64, p_min_hint=0.5)
        d = (out[:, 0] - ref[:, 0]).abs().max().item() if out.shape == ref.shape \
            else None
        print(f"  形状 {tuple(out.shape)} vs {tuple(ref.shape)}  max|Δ|={d}")
        ms = bench(lambda: scan_assoc(p, g), reps=5, warm=1)
        ms2 = bench(lambda: gated_scan(p, g, s0, chunk=64, p_min_hint=0.5),
                    reps=5, warm=1)
        print(f"  associative_scan {ms:.2f} ms  vs 分块 {ms2:.2f} ms")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()