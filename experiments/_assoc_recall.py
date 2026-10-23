"""判据: 真 SunnyNet 上加关联记忆后, MQAR 关联召回是否提升.

`_recall_matrix.py` 已用简化结构证明矩阵值状态能把 2 对召回从 6.6% 提到 90%,
本脚本把同样的形态接到真架构上, 验证在完整块内 (门控扫描 + 反馈 + 残差) 仍然成立.

用法: python -X utf8 experiments/_assoc_recall.py
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


def train_eval(model, n_pairs, steps, B, dev, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(B, n_pairs, dev)
        logits = model(x)
        loss = F.cross_entropy(logits[:, -1], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y = make_batch(256, n_pairs, dev)
        acc = float((model(x)[:, -1].argmax(-1) == y).float().mean())
    model.train()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2,4,8")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cpu")
    torch.set_num_threads(6)

    variants = [("base", {}), ("assoc", {"assoc": True, "assoc_slots": 32, "assoc_m": 16})]
    rows = []
    for n in [int(v) for v in args.pairs.split(",")]:
        r = {"pairs": n, "seq_len": 2 * n + 1}
        for name, ab in variants:
            torch.manual_seed(0)
            cfg = SunnyConfig(vocab_size=N_SYM, dim=args.dim, inner=args.dim * 2,
                              rank=args.dim // 2, layers=args.layers, ablation=ab)
            m = SunnyNet(cfg)
            t0 = time.perf_counter()
            acc = train_eval(m, n, args.steps, args.B, dev)
            r[f"{name}_acc"] = round(acc, 4)
            r[f"{name}_params"] = m.num_params()
            r[f"{name}_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
    print(json.dumps({"dim": args.dim, "layers": args.layers, "steps": args.steps,
                      "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()