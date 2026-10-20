"""核验 p.clamp(p_floor, 1.0) 是否为 no-op.

p = 1 - delta*(1-sigmoid(gate)), delta = 1/H, H in [h_min, h_max].
delta <= 1/h_min = 0.5, 故 p >= 1 - 0.5 = 0.5 > p_floor = 1/h_max.
用真实训练步采样 p 的实际 min/max, 并逐位比较 clamp 前后的前向输出.

用法: python -X utf8 experiments/_clamp_check.py [preset] [dim]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

PRESET = sys.argv[1] if len(sys.argv) > 1 else "small"
V = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

cfg = getattr(SunnyConfig, PRESET)(V) if hasattr(SunnyConfig, PRESET) else SunnyConfig.small(V)
cfg.scan_chunk = 64
m = SunnyNet(cfg).train()
opt = torch.optim.SGD(m.parameters(), lr=3e-3)

stats = {"p_min": 1.0, "p_max": 0.0, "raw_min": 1.0, "raw_max": 0.0,
         "floor": None, "bound": None}

import sunnynet.layers as L  # noqa: E402

_orig_core = L.SunnyBlock._core
for blk in m.blocks:
    stats["floor"] = float(blk.p_floor)
    stats["bound"] = (1.0 / blk.h_min, 1.0 / blk.h_max)


def recording_core(self, a):
    u, gate, p_clamped = _orig_core(self, a)
    with torch.no_grad():
        delta = self.delta().view(1, 1, -1)
        p_raw = 1.0 - delta * (1.0 - torch.sigmoid(gate))
        stats["raw_min"] = min(stats["raw_min"], float(p_raw.min()))
        stats["raw_max"] = max(stats["raw_max"], float(p_raw.max()))
        stats["p_min"] = min(stats["p_min"], float(p_clamped.min()))
        stats["p_max"] = max(stats["p_max"], float(p_clamped.max()))
    return u, gate, p_clamped


L.SunnyBlock._core = recording_core

x = torch.randint(0, V, (8, 256))
y = torch.randint(0, V, (8, 256))
for _ in range(20):
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

print(f"preset={PRESET} dim={cfg.dim} p_floor={stats['floor']:.8g} "
      f"delta_range=({stats['bound'][1]:.6g},{stats['bound'][0]:.6g})")
print(f"raw p (clamp 前) min={stats['raw_min']:.10g}  max={stats['raw_max']:.10g}")
print(f"clamped p       min={stats['p_min']:.10g}  max={stats['p_max']:.10g}")
is_noop = stats["raw_min"] >= stats["floor"] and stats["raw_max"] <= 1.0
print(f"clamp 是 no-op: {is_noop}")

# 逐位比较: 去掉 clamp 后的前向输出
y0 = m(x).detach().clone()


def core_noclamp(self, a):
    q = self.U(self.V(a))
    u_raw, gate = q.split(self.dim, dim=-1)
    u = u_raw * self._compete(gate)
    delta = self.delta().view(1, 1, -1)
    if self.use_decay_gate:
        p = 1.0 - delta * (1.0 - torch.sigmoid(gate))
    else:
        p = (1.0 - delta).expand_as(u)
    return u, gate, p


L.SunnyBlock._core = core_noclamp
y1 = m(x).detach().clone()
md = float((y0 - y1).abs().max())
print(f"去 clamp 后前向最大偏差: {md:.3e}  (fp32 应 <=1e-6)")
print("VERDICT " + ("NOOP_SAFE" if (is_noop and md <= 1e-6) else "NOT_NOOP"))