"""在固定栈参数预算下扫描 (dim, layers) 网格: 自动求解 rank, 测训练步速度.

动机: 逐元素算子按激活宽度付费, 宽度由 dim 决定; 层数则乘算整体开销.
在参数预算固定时, 二者存在权衡. 本脚本给出帕累托前沿.

用法: python -X utf8 experiments/_cfg_scan.py <dim> <layers> <target_params> [T] [B] [reps]
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

DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 128
LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 365844
INNER = int(sys.argv[4]) if len(sys.argv) > 4 else 0
T = int(sys.argv[5]) if len(sys.argv) > 5 else 512
B = int(sys.argv[6]) if len(sys.argv) > 6 else 8
REPS = int(sys.argv[7]) if len(sys.argv) > 7 else 25
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

inner = INNER if INNER > 0 else 2 * DIM


def params_of(rank):
    cfg = SunnyConfig(vocab_size=VOCAB, dim=DIM, inner=inner, rank=rank,
                      layers=LAYERS, h_max=2048.0)
    return SunnyNet(cfg).num_params(non_embedding=True)


best = None
for rank in range(16, 640, 4):
    s = params_of(rank)
    d = abs(s - TARGET)
    if best is None or d < best[0]:
        best = (d, rank, s)
    if s > TARGET * 1.2:
        break
d, rank, stack = best
bias = 100 * (stack - TARGET) / TARGET

cfg = SunnyConfig(vocab_size=VOCAB, dim=DIM, inner=inner, rank=rank,
                  layers=LAYERS, h_max=2048.0)
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
    lv = step()
t0 = time.perf_counter()
for _ in range(REPS):
    lv = step()
dt = (time.perf_counter() - t0) / REPS * 1e3

out = {"dim": DIM, "layers": LAYERS, "inner": inner, "rank": rank, "stack": stack,
       "bias_pct": round(bias, 3), "T": T, "B": B, "train_ms": round(dt, 3),
       "ms_per_token": round(dt / (B * T) * 1e3, 5), "loss": round(lv, 4)}
print(f"[dim={DIM:3d} L={LAYERS} rank={rank:3d}] stack={stack:,} ({bias:+.2f}%) "
      f"train={out['train_ms']}ms ms/tok={out['ms_per_token']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))