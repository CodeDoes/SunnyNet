"""真实训练步的组件分解: 前向 / CE / 反向 / 优化器 各自耗时, 及算子计数.

用法: python -X utf8 experiments/_step_breakdown.py [T] [reps]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
V, B = 8192, 8

torch.set_num_threads(8)
torch.manual_seed(1234)
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T))
y = torch.randint(0, V, (B, T))

print(f"T={T} B={B} V={V} stack={m.num_params(non_embedding=True):,} "
      f"threads={torch.get_num_threads()}")


def timeit(fn, warmup=3, iters=REPS):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def fwd():
    return m(x)


t_fwd = timeit(fwd)


def fwd_ce():
    return F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))


t_fwd_ce = timeit(fwd_ce)


def fwd_bwd():
    loss = fwd_ce()
    loss.backward()
    return loss


t_fwd_bwd = timeit(fwd_bwd)


def step():
    l = fwd_ce()
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()


t_step = timeit(step)
t_ce = t_fwd_ce - t_fwd
t_bwd = t_fwd_bwd - t_fwd_ce
t_opt = t_step - t_fwd_bwd

# 算子计数 (以 profiler 统计 aten 调用次数)
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
evts = prof.key_averages()
top = sorted(evts, key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
total_self = sum(e.self_cpu_time_total for e in evts)

print(f"\n前向 {t_fwd:.2f} ms | CE 单独 {t_ce:.2f} ms | 反向 {t_bwd:.2f} ms | 完整步 {t_step:.2f} ms")
print(f"前向占比 {t_fwd / t_step * 100:.1f}%  CE占比 {t_ce / t_step * 100:.1f}%  "
      f"反向占比 {t_bwd / t_step * 100:.1f}%")
n_calls = sum(e.count for e in evts)
print(f"aten 算子调用总数/步: {n_calls}")

print("\n=== self CPU 时间 top15 ===")
for e in top:
    print(f"  {e.key[:58]:58s} {e.self_cpu_time_total / 1000:8.2f} ms "
          f"{e.self_cpu_time_total / max(total_self, 1) * 100:5.1f}%  n={e.count}")

print("\nJSON " + json.dumps({"T": T, "fwd_ms": round(t_fwd, 2), "ce_ms": round(t_ce, 2),
                              "bwd_ms": round(t_bwd, 2), "step_ms": round(t_step, 2),
                              "aten_calls": n_calls}))