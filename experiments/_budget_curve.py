"""训练预算与连贯性的关系: 区分"训练不足"与"架构不佳".

外部 Kaggle 运行是 0.96 token/参数 (Chinchilla 的 1/21), 输出不连贯.
本实验回答: 在同等 token 预算下, SunnyNet 与 Transformer 的 val loss 曲线谁更好.
若两者都"不连贯"且曲线同步, 则归因是预算不足, 不是架构.

用法: python -X utf8 experiments/_budget_curve.py --arch sunnynet --tokens 4M
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB = 8192


def load_tokens(domain):
    p = ROOT / f"data/proc/tokens_{domain}.bin"
    v = ROOT / f"data/proc/val_{domain}.bin"
    tr = np.memmap(p, dtype=np.uint16, mode="r")
    va = np.memmap(v, dtype=np.uint16, mode="r")
    return torch.from_numpy(np.asarray(tr, dtype=np.int64)), \
        torch.from_numpy(np.asarray(va, dtype=np.int64))


def main():
    def _tok(v):
        v = str(v).strip().upper()
        return float(v[:-1]) * 1e6 if v.endswith("M") else (
            float(v[:-1]) * 1e9 if v.endswith("B") else float(v))

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sunnynet")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--domain", default="en_literature")
    ap.add_argument("--tokens", type=_tok, default=4.0)   # 4M
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    torch.set_num_threads(6)
    from train.train import build_models, lr_at
    tr, va = load_tokens(args.domain)
    n_tok = int(args.tokens * 1e6)
    steps = n_tok // (args.B * args.T)

    sunny, base, cfg = build_models(args.arch, VOCAB, args.preset, None, args.T, 1234)
    model = sunny if args.arch == "sunnynet" else base
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    rng = np.random.default_rng(0)
    marks = []
    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, args.lr, max(1, steps // 20))
        i = rng.integers(0, max(1, len(tr) - args.B * args.T - 2))
        x = tr[i:i + args.B * args.T].reshape(args.B, args.T)
        y = tr[i + 1:i + 1 + args.B * args.T].reshape(args.B, args.T)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, steps // 8) == 0:
            with torch.no_grad():
                vl = 0.0
                for k in range(4):
                    j = k * (args.B * args.T)
                    vx = va[j:j + args.B * args.T].reshape(args.B, args.T)
                    vy = va[j + 1:j + 1 + args.B * args.T].reshape(args.B, args.T)
                    vl += float(torch.nn.functional.cross_entropy(
                        model(vx).reshape(-1, VOCAB), vy.reshape(-1), reduction="sum"))
                vl /= 4 * args.B * args.T
            marks.append({"step": step + 1, "tok": (step + 1) * args.B * args.T,
                          "val": round(vl, 4), "ppl": round(math.exp(min(vl, 20)), 3)})
    dt = time.perf_counter() - t0

    print(json.dumps({"arch": args.arch, "preset": args.preset, "domain": args.domain,
                      "steps": steps, "tokens": steps * args.B * args.T,
                      "wall_s": round(dt, 1), "ms_token": round(dt * 1000 / (steps * args.B * args.T), 4),
                      "marks": marks}, ensure_ascii=False))


if __name__ == "__main__":
    main()