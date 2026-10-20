"""_compete 门控中 roll 的 GPU/CPU 微基准: roll vs 索引切片.

CPU 上曾实测 roll->切片更慢 (已否决); GPU 上 roll kernel 占比 3.1%/1.9%,
成本结构不同, 需分平台重测. 数值等价性在本脚本内验证.

用法: python -X utf8 experiments/_compete_bench.py [device] [T] [B]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

dev_name = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
D = 512
GROUP = 4
dev = torch.device(dev_name)


def compete_roll(gg):
    return torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
        torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))


def compete_index(gg):
    n = gg.shape[-1]
    idx1 = torch.cat([torch.arange(1, n, device=gg.device),
                      torch.arange(0, 1, device=gg.device)])
    idx2 = torch.cat([torch.arange(2, n, device=gg.device),
                      torch.arange(0, 2, device=gg.device)])
    return torch.sigmoid(gg - gg.index_select(-1, idx1)) * \
        torch.sigmoid(gg - gg.index_select(-1, idx2))


torch.manual_seed(0)
gg = torch.randn(B, T, D // GROUP, GROUP, device=dev)

r1 = compete_roll(gg)
r2 = compete_index(gg)
maxdiff = float((r1 - r2).abs().max())

gg.requires_grad_(True)
a = compete_roll(gg).sum()
a.backward()
g_roll = gg.grad.clone()
gg.grad = None
b = compete_index(gg).sum()
b.backward()
g_idx = gg.grad.clone()
grad_diff = float((g_roll - g_idx).abs().max())


def timeit(fn, n=200):
    for _ in range(20):
        fn(gg)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(gg)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


out = {"device": dev_name, "T": T, "B": B, "D": D, "group": GROUP,
       "fwd_maxdiff": maxdiff, "grad_maxdiff": grad_diff}
for label, fn in (("roll", compete_roll), ("index", compete_index)):
    out[f"fwd_{label}_us"] = round(timeit(fn), 2)
out["fwd_ratio"] = round(out["fwd_index_us"] / out["fwd_roll_us"], 3)
print(f"[{dev_name}] T={T} B={B} fwd_maxdiff={maxdiff:.2e} grad_maxdiff={grad_diff:.2e}")
print(f"  roll={out['fwd_roll_us']}us index={out['fwd_index_us']}us ratio={out['fwd_ratio']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))