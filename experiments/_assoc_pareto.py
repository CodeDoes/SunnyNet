"""关联记忆的容量-成本前沿: 小 state 下召回还剩多少.

`_assoc_profile.py` 测出 state 宽度 8192 (slots 128 x m 64) 训练慢数十倍,
`_assoc_recall.py` 在小 state (slots 32 x m 16) 上召回 96.1%.

本脚本扫 state 宽度, 给出"召回 vs 训练成本"的权衡曲线, 用于选定默认配置.

用法: python -X utf8 experiments/_assoc_pareto.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet

N_SYM = 64


def make_batch(B, n_pairs, dev):
    k = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    v = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    qi = torch.randint(0, n_pairs, (B,), device=dev)
    x = torch.stack([k, v], dim=2).reshape(B, 2 * n_pairs)
    q = k.gather(1, qi[:, None])
    x = torch.cat([x, q], dim=1)
    y = v.gather(1, qi[:, None]).squeeze(1)
    return x, y


def acc(model, n_pairs, steps, B, dev, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = make_batch(B, n_pairs, dev)
        loss = F.cross_entropy(model(x)[:, -1], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        x, y = make_batch(256, n_pairs, dev)
        a = float((model(x)[:, -1].argmax(-1) == y).float().mean())
    return a, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cpu")
    torch.set_num_threads(6)

    grid = [(0, 0), (16, 8), (32, 16), (32, 32), (64, 32)]
    rows = []
    for sl, mm in grid:
        torch.manual_seed(0)
        ab = {} if sl == 0 else {"assoc": True, "assoc_slots": sl, "assoc_m": mm}
        cfg = SunnyConfig(vocab_size=N_SYM, dim=args.dim, inner=args.dim * 2,
                          rank=args.dim // 2, layers=args.layers, ablation=ab)
        m = SunnyNet(cfg)
        a, dt = acc(m, args.pairs, args.steps, args.B, dev)
        rows.append({"slots": sl, "m": mm, "state": sl * mm,
                     "acc": round(a, 4), "train_s": round(dt, 1),
                     "params": m.num_params()})
        print(f"  slots={sl:3d} m={mm:3d} state={sl*mm:5d} "
              f"acc={a:.4f} train={dt:6.1f}s params={m.num_params():,}")
    print(json.dumps({"pairs": args.pairs, "dim": args.dim, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()