"""等宽对照: 把 sunnynet 的激活宽度压到与同行一致, 判定"宽度"是否是全部原因.

同行 (mamba/h3/rwkv) 在栈参数 365k 时 d_model=80, d_ff=160.
sunnynet 是 dim=128, inner=192 -> 逐元素算子按更宽张量付费.

本脚本给 sunnynet 一组"窄配置", 参数量仍对齐, 看速度差距是否消失.
若消失 -> 差距来自宽度 (可用配置调整解决);
若不消失 -> 差距来自算子结构本身.

用法: python -X utf8 experiments/_narrow_ab.py <name> [T] [B] [reps]
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.linear_baselines import _make, build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "sunny_base"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 25
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

SUNNY = {
    "sunny_base":   dict(dim=128, inner=192, rank=64,  layers=4),
    "sunny_narrow": dict(dim=96,  inner=160, rank=80,  layers=4),
    "sunny_narrow2": dict(dim=80, inner=160, rank=96,  layers=5),
    "sunny_narrow3": dict(dim=80, inner=120, rank=128, layers=5),
}

if NAME in SUNNY:
    v = SUNNY[NAME]
    cfg = SunnyConfig(vocab_size=VOCAB, dim=v["dim"], inner=v["inner"],
                      rank=v["rank"], layers=v["layers"], h_max=2048.0)
    cfg.scan_chunk = 64
    m = SunnyNet(cfg)
    d_act = (v["dim"], v["inner"])
else:
    ref = SunnyNet(SunnyConfig.small(VOCAB))
    target = ref.num_params(non_embedding=True)
    del ref
    c, _, _ = build_linear_baseline(NAME, target, VOCAB, max_len=max(T, 2048), n_layers=6)
    m = _make(NAME, c)
    d_act = (c.d_model, c.d_ff)

m = m.train()
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

out = {"name": NAME, "dim": d_act[0], "inner": d_act[1], "stack": stack,
       "T": T, "B": B, "train_ms": round(dt, 3),
       "ms_per_token": round(dt / (B * T) * 1e3, 5), "loss": round(lv, 4)}
print(f"[{NAME:14s}] dim={d_act[0]:3d} inner={d_act[1]:3d} stack={stack:,} "
      f"train={out['train_ms']}ms ms/tok={out['ms_per_token']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))