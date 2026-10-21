"""训练步中 scan 路径的占比: 用 profiler 统计 gated_scan 相关 aten 调用与自耗时.

用法: python -X utf8 experiments/_scan_share.py [T] [reps]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
V, B = 8192, 8
torch.set_num_threads(8)
torch.manual_seed(1234)
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T))
y = torch.randint(0, V, (B, T))


def step():
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


for _ in range(3):
    step()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    step()

evts = prof.key_averages()
total = sum(e.self_cpu_time_total for e in evts)

# scan 家族: gated_scan 内部用到的算子
SCAN_OPS = {"aten::cumsum", "aten::exp", "aten::log", "aten::log1p",
            "aten::expm1", "aten::pow", "aten::clamp", "aten::clamp_"}
scan_ms = sum(e.self_cpu_time_total for e in evts if e.key in SCAN_OPS) / 1000
scan_n = sum(e.count for e in evts if e.key in SCAN_OPS)

print(f"T={T} stack={m.num_params(non_embedding=True):,} threads=8")
print(f"profiler 总 self CPU = {total/1000:.1f} ms")
print(f"scan 家族算子 self = {scan_ms:.2f} ms ({scan_ms/(total/1000)*100:.1f}%)  n={scan_n}")
print("\n=== 全部算子按 self 排序 top25 ===")
for e in sorted(evts, key=lambda e: e.self_cpu_time_total, reverse=True)[:25]:
    print(f"  {e.key[:52]:52s} {e.self_cpu_time_total/1000:7.2f}ms "
          f"{e.self_cpu_time_total/total*100:5.1f}%  n={e.count}")
print("\nJSON " + json.dumps({"T": T, "total_self_ms": round(total/1000, 2),
                              "scan_family_ms": round(scan_ms, 2),
                              "scan_family_count": scan_n,
                              "aten_calls": sum(e.count for e in evts)}))
