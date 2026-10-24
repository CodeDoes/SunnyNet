"""定位关联记忆在 CPU 训练步上的开销构成, 用于决定优化方向.

用法: python -X utf8 experiments/_assoc_hotspot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet


def main():
    torch.set_num_threads(6)
    cfg = SunnyConfig(vocab_size=8192, dim=128, inner=192, rank=64, layers=4,
                      h_max=2048.0, ablation={"assoc": True, "assoc_slots": 32,
                                              "assoc_m": 16})
    torch.manual_seed(0)
    m = SunnyNet(cfg)
    idx = torch.randint(0, 8192, (8, 256))
    y = torch.randint(0, 8192, (8, 256))

    from torch.profiler import ProfilerActivity, profile

    def step():
        loss = torch.nn.functional.cross_entropy(m(idx).reshape(-1, 8192),
                                                 y.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(5):
            step()

    key = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18)
    print(key)

    ev = prof.key_averages()
    tot = sum(e.self_cpu_time_total for e in ev)
    assoc_hits = ("logspace_scan", "cumsum", "exp", "einsum", "bmm", "matmul")
    for name in assoc_hits:
        t = sum(e.self_cpu_time_total for e in ev if name in e.key)
        if t > 0:
            print(f"  {name:16s} {t/1000:8.1f} ms  {t/max(tot,1)*100:5.1f}%")


if __name__ == "__main__":
    main()