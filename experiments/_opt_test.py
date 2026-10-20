"""优化器与反向侧杠杆测试.

运行: python -X utf8 experiments/_opt_test.py
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
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000


def main():
    torch.set_num_threads(8)
    vocab, B, T = 8192, 8, 1024
    torch.manual_seed(0)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    ce = torch.nn.functional.cross_entropy

    def make(opt):
        def step():
            lg = m(x)
            loss = ce(lg.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return step

    def noopt():
        lg = m(x)
        loss = ce(lg.reshape(-1, vocab), y.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()

    print("=== 优化器变体 (B=8 T=1024) ===", flush=True)
    print(f"  仅前反向(不含更新)   {bench(noopt):8.1f} ms", flush=True)

    for name, kw in (("AdamW 默认", {}), ("AdamW foreach", {"foreach": True})):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, **kw)
        print(f"  {name:20s}{bench(make(opt)):8.1f} ms", flush=True)

    try:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
        print(f"  {'AdamW fused':20s}{bench(make(opt)):8.1f} ms", flush=True)
    except Exception as e:
        print(f"  AdamW fused 不可用: {type(e).__name__}: {str(e)[:80]}")

    print("\n=== 参数张量数 ===", flush=True)
    ps = list(m.parameters())
    print(f"  张量数 {len(ps)}  元素 {sum(p.numel() for p in ps):,}")
    print(f"  形状集合 {sorted(set(tuple(p.shape) for p in ps))}")


if __name__ == "__main__":
    main()