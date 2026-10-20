"""栈参数预算下的宽度配置扫描: 训练步速度 (相同参数量, 不同分配).

动机: 逐元素算子按激活宽度付费. 参数量放在 rank 通路 (激活宽 2*dim)
比放在 inner 通路 (激活宽 inner) 更"便宜". 本脚本量化该效应.

每格独立子进程. 只测速度, 质量另做 A/B.

用法: python -X utf8 experiments/_width_sweep.py <name> [T] [B] [reps]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

VARIANTS = {
    "base":   dict(dim=128, inner=192, rank=64, layers=4),
    "A_wide_rank": dict(dim=128, inner=128, rank=112, layers=4),
    "B_narrow_inner": dict(dim=128, inner=96, rank=128, layers=4),
    "C_more_layers": dict(dim=96, inner=160, rank=64, layers=6),
    "D_fewer_layers": dict(dim=128, inner=256, rank=96, layers=3),
    "E_wide_rank_5l": dict(dim=128, inner=112, rank=80, layers=5),
}
NAME = sys.argv[1] if len(sys.argv) > 1 else "base"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 25
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

v = VARIANTS[NAME]
cfg = SunnyConfig(vocab_size=VOCAB, dim=v["dim"], inner=v["inner"], rank=v["rank"],
                  layers=v["layers"], h_max=2048.0)
cfg.scan_chunk = 64
m = SunnyNet(cfg).train()
stack = m.num_params(non_embedding=True)
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
    lv = step()
t0 = time.perf_counter()
for _ in range(REPS):
    lv = step()
dt = (time.perf_counter() - t0) / REPS * 1e3

out = {"name": NAME, "config": v, "T": T, "B": B, "stack": stack,
       "train_ms": round(dt, 3), "ms_per_token": round(dt / (B * T) * 1e3, 5),
       "loss": round(lv, 4)}
print(f"[{NAME:16s}] stack={stack:,} train={out['train_ms']}ms "
      f"ms/tok={out['ms_per_token']} loss={out['loss']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))