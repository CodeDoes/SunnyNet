"""端到端判决: MoE 稀疏 MLP 的红利能否穿透到真实模型的前向+反向.

算子级的稀疏红利 (CPU 3.47x / GPU 0.75x) 未必能在端到端兑现 —— 本仓库已有两次
"算子快、端到端无差"的前例. 故必须用真实 token 跑前向+反向, 同参数同有效 FLOPs.

结构: norm -> 递推块(用线性 attention 近似) -> MLP(稠密 or E 专家只走 1) -> norm -> head.
判据: ms/token 与 ppl. 相同 ppl 下 CPU 的 MoE 更省时间才算数.

用法: python -X utf8 experiments/_sparse_e2e.py --device cpu
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB = 8192


class Recur(nn.Module):
    """线性因果递推: s_t = a*s_{t-1} + (1-a)*x_t, 每通道标量 a."""

    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.a = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        h = self.n(x)
        a = torch.sigmoid(self.a).view(1, 1, -1)
        B, T, d = h.shape
        s = torch.zeros(B, d, device=h.device, dtype=h.dtype)
        outs = []
        for t in range(T):
            s = a[:, 0] * s + (1 - a[:, 0]) * h[:, t]
            outs.append(s)
        return x + torch.stack(outs, dim=1)


class DenseMLP(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.f = nn.Linear(d, d * mult, bias=False)
        self.p = nn.Linear(d * mult, d, bias=False)

    def forward(self, x):
        return x + self.p(F.gelu(self.f(self.n(x))))


class MoEMLP(nn.Module):
    """E 个等大专家, 每 token 走 top-1. 有效 FLOPs = 稠密 / E 的 f 部分."""

    def __init__(self, d, n_expert=8, mult=4):
        super().__init__()
        self.E, self.d = n_expert, d
        self.n = nn.LayerNorm(d)
        self.f = nn.Linear(d, d * mult, bias=False)     # 总宽与稠密相同
        self.p = nn.Linear(d * mult, d, bias=False)
        self.gate = nn.Linear(d, n_expert, bias=False)
        self.h = d * mult // n_expert

    def forward(self, x):
        B, S, d = x.shape
        hx = self.n(x)
        N = B * S
        fx = hx.reshape(N, d)
        g = F.softmax(self.gate(fx), dim=-1)
        topi = g.argmax(-1)
        wgt = g.gather(1, topi[:, None])
        out = torch.zeros(N, d, device=x.device, dtype=x.dtype)
        h = self.h
        for e in range(self.E):
            rows = (topi == e).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            he = F.gelu(fx[rows] @ self.f.weight[e * h:(e + 1) * h].t())
            oe = he @ self.p.weight[:, e * h:(e + 1) * h].t()
            out.index_add_(0, rows, oe * wgt[rows])
        return x + out.reshape(B, S, d)


class LM(nn.Module):
    def __init__(self, vocab, d, layers, kind, n_expert=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        blocks = []
        for _ in range(layers):
            blocks.append(Recur(d))
            blocks.append(DenseMLP(d) if kind == "dense" else MoEMLP(d, n_expert))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def load_tokens(n):
    p = ROOT / "data/proc/tokens_en_literature.bin"
    toks = np.memmap(p, dtype=np.uint16, mode="r")
    return torch.from_numpy(np.asarray(toks[:n], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)
    toks = load_tokens(args.B * args.T * (args.steps + 8) + 8)
    rows = []
    for kind in ("dense", "moe"):
        torch.manual_seed(0)
        m = LM(VOCAB, args.d, args.layers, kind, args.experts).to(dev)
        nparam = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
        idx = 0

        def step():
            nonlocal idx
            x = toks[idx:idx + args.B * args.T].reshape(args.B, args.T).to(dev)
            y = toks[idx + 1:idx + 1 + args.B * args.T].reshape(args.B, args.T).to(dev)
            idx += args.B * args.T
            logits = m(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            return float(loss)

        for _ in range(5):
            step()
        t0 = time.perf_counter()
        losses = [step() for _ in range(args.steps)]
        dt = time.perf_counter() - t0
        rows.append({
            "kind": kind, "params": nparam,
            "ms_step": round(dt / args.steps * 1000, 2),
            "ms_token": round(dt / args.steps * 1000 / (args.B * args.T), 5),
            "loss_last": round(sum(losses[-5:]) / 5, 4),
        })

    d = rows[0]
    m = rows[1]
    out = {"device": args.device, "d": args.d, "layers": args.layers,
           "B": args.B, "T": args.T, "steps": args.steps, "rows": rows,
           "e2e_dividend": round(d["ms_token"] / m["ms_token"], 3),
           "param_ratio": round(m["params"] / d["params"], 3)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()