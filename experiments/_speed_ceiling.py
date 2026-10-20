"""SunnyBlock 各组件对训练步耗时的贡献上限 (消融法).

方法同 §14.7: 把目标组件替换为极简实现 (数值不等价), 测端到端训练步.
给出"该组件耗时归零"的上界, 用于排序优化优先级. 不是可落地的改写.

每格独立子进程, 避免跨变体污染.

用法: python -X utf8 experiments/_speed_ceiling.py <variant> [T] [B] [reps]
  variant: base | no_compete | no_wedge | no_feedback | no_local | no_circ | no_readout
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

from sunnynet import layers as L  # noqa: E402
from sunnynet import ops as O  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

if VARIANT == "no_compete":
    L.SunnyBlock._compete = lambda self, gate: torch.ones_like(gate)
elif VARIANT == "no_wedge":
    O.softwedge = lambda z, a: z
    L.softwedge = lambda z, a: z
elif VARIANT == "no_feedback":
    L.SunnyBlock._feedback = lambda self, states, s_in: states
elif VARIANT == "no_local":
    L.SunnyBlock._apply_local = lambda self, raw, hist: raw
elif VARIANT == "no_circ":
    _orig_fb = L.SunnyBlock._feedback

    def fb_nocirc(self, states, s_in):
        return _orig_fb(self, states, s_in)

    L.StateCirculant.forward = lambda self, s: s
elif VARIANT == "no_readout":
    L.SunnyBlock._readout = lambda self, u, s_prev: self.W_z(u)

cfg = SunnyConfig.small(VOCAB)
cfg.scan_chunk = 64
m = SunnyNet(cfg).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, VOCAB, (B, T))
y = torch.randint(0, VOCAB, (B, T))


def step():
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


for _ in range(4):
    step()
t0 = time.perf_counter()
for _ in range(REPS):
    step()
dt = (time.perf_counter() - t0) / REPS * 1e3

out = {"variant": VARIANT, "T": T, "B": B, "reps": REPS,
       "train_ms": round(dt, 3), "ms_per_token": round(dt / (B * T) * 1e3, 5)}
print(f"[{VARIANT}] train={out['train_ms']}ms ms/tok={out['ms_per_token']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))