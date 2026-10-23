"""MQAR: 元素级状态 vs 逐 head 矩阵值状态, 同参数预算对照.

`_recall_capacity.py` 测出元素级递推在 2 对关联召回上仅 4.69% (注意力 93.36%),
根因是每层只有 dim 个标量的记忆格. 本脚本判定升级到矩阵值状态能否补上这块容量:
每 head 存一个 (m, m) 外积累加器, 读出 o = S^T q.

凸组合形式保持有界:
    S_t = p ⊙ S_{t-1} + (1 - p) ⊙ (v_t ⊗ k_t)
对任意 p ∈ [0,1] 有 |S_t| <= max(|S_0|, |v ⊗ k|) 逐元素成立.

用法: python -X utf8 experiments/_recall_matrix.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_SYM = 64


def make_batch(B, n_pairs, dev):
    """序列 = k1 v1 ... kn vn kq, 目标是 v_q."""
    k = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    v = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    qi = torch.randint(0, n_pairs, (B,), device=dev)
    x = torch.stack([k, v], dim=2).reshape(B, 2 * n_pairs)
    q = k.gather(1, qi[:, None])
    x = torch.cat([x, q], dim=1)
    y = v.gather(1, qi[:, None]).squeeze(1)
    return x, y


def nparams(m):
    return sum(p.numel() for p in m.parameters())


class ElemRecall(nn.Module):
    """元素级凸组合: 状态宽度 = d, 与 n_pairs 无关."""

    def __init__(self, d=64, n_layer=2, n_head=4):
        super().__init__()
        self.emb = nn.Embedding(N_SYM, d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.alpha = nn.Parameter(torch.zeros(n_layer, d))
        self.head = nn.Linear(d, N_SYM)

    def forward(self, x):
        h = self.emb(x)
        B, T, d = h.shape
        s = [torch.zeros(B, d, device=x.device) for _ in self.layers]
        out = []
        for t in range(T):
            ht = h[:, t]
            for i, li in enumerate(self.layers):
                a = torch.sigmoid(self.alpha[i])
                s[i] = a * s[i] + (1 - a) * torch.tanh(li(ht))
                ht = s[i]
            out.append(ht)
        return self.head(torch.stack(out, 1))


class MatrixRecall(nn.Module):
    """逐 head (m, m) 矩阵值状态, 凸组合注入, 读出 o = S^T q."""

    def __init__(self, d=64, n_layer=2, n_head=4):
        super().__init__()
        m = d // n_head
        self.d, self.h, self.m = d, n_head, m
        self.emb = nn.Embedding(N_SYM, d)
        self.Wk = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.Wv = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.Wq = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.alpha = nn.Parameter(torch.zeros(n_layer, n_head))
        self.n_layer = n_layer
        self.head = nn.Linear(d, N_SYM)

    def forward(self, x):
        h = self.emb(x)
        B, T, _ = h.shape
        H, m = self.h, self.m
        S = [torch.zeros(B, H, m, m, device=x.device) for _ in range(self.n_layer)]
        out = []
        for t in range(T):
            ht = h[:, t]
            for i in range(self.n_layer):
                k = F.normalize(self.Wk[i](ht).view(B, H, m), dim=-1)
                v = self.Wv[i](ht).view(B, H, m)
                q = F.normalize(self.Wq[i](ht).view(B, H, m), dim=-1)
                a = torch.sigmoid(self.alpha[i]).view(1, H, 1, 1)
                S[i] = a * S[i] + (1 - a) * (v.unsqueeze(-1) * k.unsqueeze(-2))
                o = (S[i] @ q.unsqueeze(-1)).squeeze(-1).reshape(B, self.d)
                ht = ht + self.norm[i](o)
            out.append(ht)
        return self.head(torch.stack(out, 1))


class DeltaRecall(nn.Module):
    """逐 head 矩阵状态 + rank-1 正交校正 (delta rule).

    先按当前键读出去掉的旧关联, 再写入新关联:
        S_t = S_{t-1} - beta*(S_{t-1} k) ⊗ k + beta*(v ⊗ k)
    把已有键对应的格子清掉再覆盖, 不像纯外积累加那样只剩衰减可用.
    """

    def __init__(self, d=64, n_layer=2, n_head=4):
        super().__init__()
        m = d // n_head
        self.d, self.h, self.m = d, n_head, m
        self.emb = nn.Embedding(N_SYM, d)
        self.Wk = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.Wv = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.Wq = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layer)])
        self.Wb = nn.ModuleList([nn.Linear(d, n_head) for _ in range(n_layer)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.n_layer = n_layer
        self.head = nn.Linear(d, N_SYM)

    def forward(self, x):
        h = self.emb(x)
        B, T, _ = h.shape
        H, m = self.h, self.m
        S = [torch.zeros(B, H, m, m, device=x.device) for _ in range(self.n_layer)]
        out = []
        for t in range(T):
            ht = h[:, t]
            for i in range(self.n_layer):
                k = F.normalize(self.Wk[i](ht).view(B, H, m), dim=-1)
                v = self.Wv[i](ht).view(B, H, m)
                q = F.normalize(self.Wq[i](ht).view(B, H, m), dim=-1)
                beta = torch.sigmoid(self.Wb[i](ht)).view(B, H, 1)
                old = torch.einsum("bhkv,bhk->bhv", S[i], k)
                S[i] = S[i] - beta.unsqueeze(-1) * (k.unsqueeze(-1) * old.unsqueeze(-2)) \
                    + beta.unsqueeze(-1) * (v.unsqueeze(-1) * k.unsqueeze(-2))
                o = torch.einsum("bhkv,bhk->bhv", S[i], q).reshape(B, self.d)
                ht = ht + self.norm[i](o)
            out.append(ht)
        return self.head(torch.stack(out, 1))


def train_eval(model, n_pairs, steps, B, dev, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(B, n_pairs, dev)
        loss = F.cross_entropy(model(x)[:, -1], y)
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
    ap.add_argument("--pairs", default="2,4,8,16")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--elem-layers", type=int, default=2)
    ap.add_argument("--mat-layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device("cpu")
    torch.set_num_threads(6)

    cfgs = [
        ("elem", ElemRecall, {"n_layer": args.elem_layers}),
        ("matrix", MatrixRecall, {"n_layer": args.mat_layers, "n_head": args.heads}),
        ("delta", DeltaRecall, {"n_layer": args.mat_layers, "n_head": args.heads}),
    ]
    rows = []
    for n in [int(v) for v in args.pairs.split(",")]:
        r = {"pairs": n, "seq_len": 2 * n + 1}
        for name, mk, kw in cfgs:
            torch.manual_seed(0)
            m = mk(args.d, **kw)
            t0 = time.perf_counter()
            acc = train_eval(m, n, args.steps, args.B, dev)
            r[f"{name}_acc"] = round(acc, 4)
            r[f"{name}_params"] = nparams(m)
            r[f"{name}_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
    print(json.dumps({"d": args.d, "steps": args.steps, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()