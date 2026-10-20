"""SunnyBlock.forward 逐段计时, 定位训练步里最重的一段.

对比同 d_model 下各段耗时, 找出与同行差异最大的环节.
张量在计时循环外构造, 避免掩盖信号.

用法: python -X utf8 experiments/_block_breakdown.py [T] [B] [reps]
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
from sunnynet.ops import logspace_scan  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
VOCAB = 8192
torch.set_num_threads(8)
torch.manual_seed(0)

cfg = SunnyConfig.small(VOCAB)
cfg.scan_chunk = 64
m = SunnyNet(cfg).train()
blk = m.blocks[0]
D = cfg.dim

x = torch.randn(B, T, D, requires_grad=True)
hist = torch.zeros(B, cfg.taps - 1, D)
s0 = torch.zeros(B, D)
x_in = torch.randn(B, T, D)

chunk = min(cfg.scan_chunk, T)


def bench(name, fn, backward=True, n=REPS):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e3
    return round(dt, 3)


res = {}

res["norm_in"] = bench("norm_in", lambda: blk.norm_in(x))
raw = blk.norm_in(x.detach()).detach()
res["apply_local"] = bench("apply_local", lambda: blk._apply_local(raw, hist))
a = blk._apply_local(raw, hist).detach()
res["core(_compete+gate)"] = bench("core", lambda: blk._core(a))
u, gate, p = blk._core(a)
u, gate, p = u.detach(), gate.detach(), p.detach()
res["scan"] = bench("scan", lambda: logspace_scan(u, p, s0, chunk=chunk,
                                                  p_min_hint=blk.p_min_bound))
states, _ = logspace_scan(u, p, s0, chunk=chunk, p_min_hint=blk.p_min_bound)
states = states.detach()
res["feedback(cat+circ)"] = bench("feedback", lambda: blk._feedback(states, s0))
res["readout(Wz+wedge+norm)"] = bench("readout", lambda: blk._readout(u, None))
y = blk._readout(u, None).detach()
res["emit(Wc+Ws)"] = bench("emit", lambda: blk.W_c(y) + blk.W_s(blk._feedback(states, s0)))
res["FULL block fwd"] = bench("full", lambda: blk(x_in, s0, hist))
xr = x_in.detach().requires_grad_(True)
res["FULL block fwd+bwd"] = bench("fullbwd", lambda: blk(xr, s0, hist)[0].sum().backward(),
                                  backward=False) if False else bench(
    "fullbwd", lambda: xr.grad.__class__ and (xr.grad is None or xr.grad.zero_()) or
    blk(xr, s0, hist)[0].sum().backward(), backward=False)

# 单层训练步 (前向+反向) 与拆解
total = sum(v for k, v in res.items() if k not in ("FULL block fwd", "FULL block fwd+bwd"))
res["_sum_parts"] = round(total, 3)
for k, v in res.items():
    print(f"  {k:26s} {v}")
print("RESULT " + json.dumps({"T": T, "B": B, "dim": D, "chunk": chunk, **res},
                             ensure_ascii=False))