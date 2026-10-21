"""解码步的算子构成剖析: 定位推理态的真实开销来源.

用法: python -X utf8 experiments/_decode_profile.py [T] [kernel_on]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
import sunnynet.kernels.core_auto as ca  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ON = (sys.argv[2] == "1") if len(sys.argv) > 2 else False
B, V = 1, 8192
torch.set_num_threads(8); torch.manual_seed(1234)
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).eval()
ca.set_kernel_active(ON, threads=8)
idx = torch.randint(0, V, (B, T))

def run(n):
    st = m.init_state(B)
    with torch.no_grad():
        for t in range(n):
            lg, st = m.forward_step(idx[:, t], st)
run(8)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    run(32)
evts = prof.key_averages()
total = sum(e.self_cpu_time_total for e in evts)
calls = sum(e.count for e in evts)
print(f"T={T} B={B} kernel={ON}  32 步总 self CPU = {total/1000:.1f} ms  ({total/1000/32:.3f} ms/步)")
print(f"算子调用总数/步 = {calls/32:.0f}")
for e in sorted(evts, key=lambda e: e.self_cpu_time_total, reverse=True)[:16]:
    print(f"  {e.key[:50]:50s} {e.self_cpu_time_total/1000/32*1000:8.3f} us/步 "
          f"{e.self_cpu_time_total/total*100:5.1f}%  n/步={e.count/32:.1f}")
