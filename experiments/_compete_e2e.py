"""_compete 中 roll vs index_select 的端到端 A/B (进程隔离).

微基准显示 index_select 更快且数值等价 (maxdiff=0), 但旧 CPU 结论是"roll 更优".
差异需在端到端训练步上判定, 微基准可能失真.

用法: python -X utf8 experiments/_compete_e2e.py <device> <impl> [T] [B] [reps]
  impl: roll | index
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet import layers as L  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

dev_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
impl = sys.argv[2] if len(sys.argv) > 2 else "roll"
T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
B = int(sys.argv[4]) if len(sys.argv) > 4 else 8
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 30
VOCAB = 8192


def compete_roll(self, gate):
    B_, T_, D_ = gate.shape
    gg = gate.view(B_, T_, D_ // self.group, self.group)
    w = torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
        torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))
    return w.reshape(B_, T_, D_)


def compete_index(self, gate):
    B_, T_, D_ = gate.shape
    gg = gate.view(B_, T_, D_ // self.group, self.group)
    n = gg.shape[-1]
    dev = gg.device
    idx1 = torch.cat([torch.arange(1, n, device=dev), torch.arange(0, 1, device=dev)])
    idx2 = torch.cat([torch.arange(2, n, device=dev), torch.arange(0, 2, device=dev)])
    w = torch.sigmoid(gg - gg.index_select(-1, idx1)) * \
        torch.sigmoid(gg - gg.index_select(-1, idx2))
    return w.reshape(B_, T_, D_)


# 找挂载 _compete 的类
target_cls = None
for name in dir(L):
    obj = getattr(L, name)
    if isinstance(obj, type) and hasattr(obj, "_compete"):
        target_cls = obj
        break
if target_cls is None:
    print("RESULT " + json.dumps({"error": "未找到 _compete 宿主类"}))
    sys.exit(1)

if impl == "index":
    target_cls._compete = compete_index

dev = torch.device(dev_name)
torch.manual_seed(1234)
cfg = SunnyConfig.small(VOCAB)
cfg.scan_chunk = 64
m = SunnyNet(cfg).to(dev).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-3)

x = torch.randint(0, VOCAB, (B, T), device=dev)
y = torch.randint(0, VOCAB, (B, T), device=dev)


def one_step():
    lg = m(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


for _ in range(5):
    one_step()
if dev.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(REPS):
    loss_v = one_step()
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / REPS * 1e3

out = {"device": dev_name, "impl": impl, "T": T, "B": B, "reps": REPS,
       "host": target_cls.__name__, "step_ms": round(dt, 3),
       "ms_per_token": round(dt / (B * T) * 1e3, 5), "loss": round(loss_v, 4)}
print(f"[{dev_name}/{impl}] step={out['step_ms']}ms ms/tok={out['ms_per_token']} loss={out['loss']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))