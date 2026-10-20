"""等价优化的严格 A/B: 用 LEGACY=1 猴子补丁复现旧实现, 独立子进程交错测量.

变体 (均由本脚本注入, 不改动 sunnynet 源码):
  LEGACY=1  -> 旧实现: cat+conv 全序列 / W_s(circ(prev)) / softwedge 含常数项
  LEGACY=0  -> 当前实现: 免 cat 快速路径 / W_s 折叠 / softwedge 去常数
两个变体数值应等价 (loss 相同), 只比耗时.

用法: LEGACY=<0|1> python -X utf8 experiments/_ab_equiv.py [T] [B] [reps]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

LEGACY = os.environ.get("LEGACY", "0") == "1"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

import torch.nn.functional as F  # noqa: E402

from sunnynet import layers as L  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

if LEGACY:
    def _apply_local_legacy(self, raw, hist):
        if self.dw is None:
            return raw
        k = self.taps - 1
        rf = torch.cat([hist, raw], dim=1) if k else raw
        return (rf + self.dw(rf))[:, k:]

    def _emit_legacy(self, u, states, s_in):
        y = self._readout(u, None)
        sn = self.norm_s(states) if self.norm_s is not None else states
        prev = self._prev(states, s_in)
        core = self.W_c(y) + self.b_gain * sn + self.W_s(self.circ(prev))
        return core

    def _readout_legacy(self, u, s_prev):
        z = self.W_z(u)
        a = self.a.tanh()
        return self.norm_mix(0.5 + 0.5 * (z / (1.0 + z.abs())) * a + z)

    L.SunnyBlock._apply_local = _apply_local_legacy
    L.SunnyBlock._emit = _emit_legacy
    L.SunnyBlock._readout = _readout_legacy

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
    return float(loss.detach())


for _ in range(5):
    loss_v = step()
t0 = time.perf_counter()
for _ in range(REPS):
    loss_v = step()
dt = (time.perf_counter() - t0) / REPS * 1e3

out = {"legacy": LEGACY, "T": T, "B": B, "reps": REPS,
       "train_ms": round(dt, 3), "ms_per_token": round(dt / (B * T) * 1e3, 5),
       "loss": round(loss_v, 4)}
print(f"[{'LEGACY' if LEGACY else 'OPT   '}] train={out['train_ms']}ms "
      f"ms/tok={out['ms_per_token']} loss={out['loss']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))