"""量化"扫描 / 竞争门控完全融合"的端到端收益上界.

做法: 分别把 _compete 与 logspace_scan 替换为极简实现 (数值不等价),
测训练步提速. 这给出"这两段耗时归零"时的收益上限, 不是可落地的改写.

用法: python -X utf8 experiments/_fuse_ceiling.py <variant> <T>
variant: base | no_compete | no_scan | both
每格独立子进程 (测量纪律).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import sunnynet.layers as L  # noqa: E402
import sunnynet.ops as O  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def main():
    variant = sys.argv[1]
    T = int(sys.argv[2])
    torch.set_num_threads(8)
    torch.manual_seed(0)
    V = 8192

    if variant in ("no_compete", "both"):
        # 竞争门控 -> 常数 (省 roll*2 + sub*2 + sigmoid*2 + mul)
        L.SunnyBlock._compete = lambda self, gate: gate.new_full(gate.shape, 0.5)

    if variant in ("no_scan", "both"):
        # 扫描 -> 单次乘加 (省 cumsum*2 + exp*2 + log + 块间循环 + stack)
        def _fake_scan(nu, p, s0, chunk=64, p_min_hint=None):
            g = (1.0 - p) * nu
            s = g.cumsum(dim=-2)
            return s, s[..., -1, :]
        O.logspace_scan = _fake_scan
        L.logspace_scan = _fake_scan

    m = SunnyNet(SunnyConfig.small(V)).train()
    x = torch.randint(0, V, (8, T))
    y = torch.randint(0, V, (8, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    print(json.dumps({
        "variant": variant, "T": T,
        "ms_per_tok": round(med * 1000 / (8 * T), 4),
        "iqr_ratio": round((ts[-1] - ts[0]) / med, 3),
    }))


if __name__ == "__main__":
    main()