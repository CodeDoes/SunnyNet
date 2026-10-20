"""在 365,844 栈参数预算下搜索最快且质量无损的 sunnynet 配置.

已知: 逐元素算子按激活宽度付费; 参数放 rank 通路比放 inner 通路便宜.
本脚本在相同栈参数下扫 (dim, inner, rank, layers) 组合, 输出速度排名.

每格独立子进程. 质量需另做 A/B (本脚本只测速度).

用法: python -X utf8 experiments/_cfg_search.py <name> [T] [B] [reps]
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

# 目标栈参数 365,844 (small 预设). 所有候选都往这个预算靠.
CANDS = {
    "base":        dict(dim=128, inner=192, rank=64,  layers=4),
    "i144_r96":    dict(dim=128, inner=144, rank=96,  layers=4),
    "i128_r108":   dict(dim=128, inner=128, rank=108, layers=4),
    "i160_r84":    dict(dim=128, inner=160, rank=84,  layers=4),
    "i144_r96_l5": dict(dim=128, inner=144, rank=96,  layers=5),
    "i144_i176_rank": dict(dim=128, inner=144, rank=88, layers=4),
}
NAME = sys.argv[1] if len(sys.argv) > 1 else "base"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 25
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(1234)

v = CANDS[NAME]
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

out = {"name": NAME, "config": v, "stack": stack, "T": T, "B": B,
       "train_ms": round(dt, 3), "ms_per_token": round(dt / (B * T) * 1e3, 5),
       "loss": round(lv, 4)}
print(f"[{NAME:14s}] dim={v['dim']:3d} inner={v['inner']:3d} rank={v['rank']:3d} "
      f"L={v['layers']} stack={stack:,} train={out['train_ms']}ms "
      f"ms/tok={out['ms_per_token']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))