"""关联召回 (MQAR): 判定"不连贯"的架构根因.

不连贯有两种来源, 必须分开:
 (a) 训练预算不足 -> 加大数据即可改善;
 (b) 固定状态的记忆容量有界 -> 长距离精确召回做不到, 加大数据无效.

MQAR 是最标准的判据: 序列 [k1 v1 k2 v2 ... kn vn] + 查询 k_i -> 预测 v_i.
状态越小, 可召回的对数越少. 注意力无此上限.

用同一套参数预算跑两种结构, 扫 对数 n 看准确率拐点.

用法: python -X utf8 experiments/_recall_capacity.py
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

N_SYM = 64          # 键/值词表大小


def make_batch(B, n_pairs, dev):
    """返回 (x, y): 序列 = k1 v1 ... kn vn kq, 目标是 v_q."""
    k = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    v = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    qi = torch.randint(0, n_pairs, (B,), device=dev)
    x = torch.stack([k, v], dim=2).reshape(B, 2 * n_pairs)          # k1 v1 k2 v2 ...
    q = k.gather(1, qi[:, None])                                     # 查询键
    x = torch.cat([x, q], dim=1)
    y = v.gather(1, qi[:, None]).squeeze(1)
    return x, y


class RecurRecall(nn.Module):
    """固定状态递推: 状态宽度 = d, 与 n_pairs 无关."""

    def __init__(self, d=64, n_layer=2):
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


class AttnRecall(nn.Module):
    """注意力: 状态随 T 增长 (KV), 可任意精确召回."""

    def __init__(self, d=64, n_layer=2, n_head=4):
        super().__init__()
        self.emb = nn.Embedding(N_SYM, d)
        self.blk = nn.ModuleList([
            nn.MultiheadAttention(d, n_head, batch_first=True) for _ in range(n_layer)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layer)])
        self.head = nn.Linear(d, N_SYM)

    def forward(self, x):
        h = self.emb(x)
        T = h.shape[1]
        m = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for att, ln in zip(self.blk, self.ln):
            o, _ = att(h, h, h, attn_mask=m, need_weights=False)
            h = ln(h + o)
        return self.head(h)


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
    ap.add_argument("--pairs", default="2,4,8,16,32")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device("cpu")
    torch.set_num_threads(6)

    rows = []
    for n in [int(v) for v in args.pairs.split(",")]:
        r = {"pairs": n, "seq_len": 2 * n + 1}
        for name, mk in (("recur", RecurRecall), ("attn", AttnRecall)):
            torch.manual_seed(0)
            m = mk(args.d)
            if name == "recur":
                npar = sum(p.numel() for p in m.parameters())
            else:
                npar = sum(p.numel() for p in m.parameters())
            t0 = time.perf_counter()
            acc = train_eval(m, n, args.steps, args.B, dev)
            r[f"{name}_acc"] = round(acc, 4)
            r[f"{name}_params"] = npar
            r[f"{name}_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)

    print(json.dumps({"d": args.d, "steps": args.steps, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()