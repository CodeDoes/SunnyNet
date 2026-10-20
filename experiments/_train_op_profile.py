"""训练步算子级剖析: 按 aten op 聚合 CPU 时间, 并对比 sunnynet 与同行.

回答"训练速度差距在哪": 是算子数量、还是单个算子慢.
每格独立子进程, 避免跨架构污染.

用法: python -X utf8 experiments/_train_op_profile.py <arch> [T] [B] [reps]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.baseline import TransformerLM, solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import _make, build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

ARCH = sys.argv[1] if len(sys.argv) > 1 else "sunnynet"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
VOCAB = 8192

torch.set_num_threads(8)
torch.manual_seed(1234)


def build():
    if ARCH == "sunnynet":
        cfg = SunnyConfig.small(VOCAB)
        cfg.scan_chunk = 64
        return SunnyNet(cfg)
    ref = SunnyNet(SunnyConfig.small(VOCAB))
    target = ref.num_params(non_embedding=True)
    del ref
    if ARCH == "transformer":
        bcfg, _, _ = solve_baseline_for_params(target, VOCAB, n_layers=6, n_heads=8,
                                               max_len=max(T, 2048))
        return TransformerLM(bcfg)
    cfg, _, _ = build_linear_baseline(ARCH, target, VOCAB, max_len=max(T, 2048),
                                      n_layers=6)
    return _make(ARCH, cfg)


m = build()
n_par = m.num_params() if hasattr(m, "num_params") else sum(p.numel() for p in m.parameters())
m.train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)

x = torch.randint(0, VOCAB, (B, T))
y = torch.randint(0, VOCAB, (B, T))


def one_step():
    lg = m(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


for _ in range(4):
    one_step()

from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(REPS):
        one_step()

rows = defaultdict(lambda: [0.0, 0])
for ev in prof.key_averages():
    cpu = getattr(ev, "self_cpu_time_total", 0.0) or 0.0
    if cpu <= 0:
        continue
    key = ev.key
    rows[key][0] += cpu
    rows[key][1] += ev.count

total = sum(v[0] for v in rows.values()) or 1.0
top = sorted([{"op": k, "cpu_ms": round(v[0] / 1000, 3),
               "pct": round(100 * v[0] / total, 2), "calls": v[1]}
              for k, v in rows.items()], key=lambda r: -r["cpu_ms"])[:18]

out = {"arch": ARCH, "T": T, "B": B, "reps": REPS, "params": n_par,
       "total_cpu_ms": round(total / 1000, 3),
       "total_calls": sum(v[1] for v in rows.values()),
       "ms_per_token": round(total / 1000 / REPS / (B * T) * 1000, 5),
       "top": top}
print(f"[{ARCH}] T={T} B={B} params={n_par:,} total={out['total_cpu_ms']}ms "
      f"calls={out['total_calls']} ms/tok={out['ms_per_token']}")
for r in top[:12]:
    print(f"  {r['op'][:44]:46s} {r['cpu_ms']:8.2f}ms {r['pct']:5.1f}% x{r['calls']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))