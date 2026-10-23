"""稀疏/条件计算的设备亲和性: CPU 与 GPU 在 top-k MLP 下的相对位置.

假设: CPU 有乱序执行与廉价分支预测, gather 便宜; GPU 走 SIMT, 分支发散与
不规则访存惩罚重. 若成立, 稀疏化会让 CPU 相对 GPU 变好 (倍数收窄), 这是
"CPU 唯一结构上占优"的候选路线.

判据: 等有效 FLOPs 下, 测 CPU 与 GPU 的 (稀疏/稠密) 耗时比.
稀疏比稠密越快 -> 该设备越适合稀疏.

用法: python -X utf8 experiments/_sparse_affinity.py --device cpu|cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def dense_mlp(x, w1, w2, b1, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2


def sparse_mlp(x, w1, w2, b1, b2, k):
    """top-k 逐样本动态路由 (最需要对 GPU 不友好的形式)."""
    logits = x @ w1 + b1
    topv, topi = torch.topk(logits, k, dim=-1)
    act = torch.relu(topv)
    B, S, kk = act.shape
    gather_w2 = w2[topi.reshape(B * S, kk)]              # (B*S, k, d)
    out = torch.bmm(act.reshape(B * S, 1, kk), gather_w2).reshape(B, S, -1)
    return out + b2


def moe_mlp(x, w1, w2, b1, b2, n_expert, per_tok):
    """MoE 式整块路由: token 按专家分批, 每批一次稠密 GEMM.

    这是真正会落地的稀疏形式 —— 无逐样本 gather, 走 batched gemm.
    per_tok = 每个 token 分到的专家数.
    """
    B, S, d = x.shape
    h = w1.shape[1] // n_expert
    logits = x @ w1.view(d, n_expert, h).sum(-1) + b1.view(n_expert, h).sum(-1)
    gate = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(gate, per_tok, dim=-1)       # (B,S,per_tok)
    N = B * S
    flat_x = x.reshape(N, d)
    topi = topi.reshape(N, per_tok)
    wgt = topv.reshape(N, per_tok)
    out = torch.zeros(N, d, device=x.device)
    for e in range(n_expert):
        rows, cols = (topi == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = flat_x[rows]
        he = torch.relu(xe @ w1[:, e * h:(e + 1) * h] + b1[e * h:(e + 1) * h])
        oe = he @ w2[e * h:(e + 1) * h] + b2
        out.index_add_(0, rows, oe * wgt[rows, cols].unsqueeze(-1))
    return out.reshape(B, S, d)


def sparse_weight_mm(x, w1, w2, b1, b2, keep_frac):
    """结构化块稀疏权重: 走 MKL 稀疏矩阵乘, 权重常驻稀疏格式."""
    d, h = w1.shape
    mask = torch.rand(h, device=w1.device) < keep_frac
    w1s = w1 * mask
    sp1 = w1s.t().to_sparse_csr()
    hx = torch.relu(torch.sparse.mm(sp1, x.reshape(-1, d).t()).t() + b1)
    out = hx @ w2 + b2
    return out.reshape(x.shape[0], x.shape[1], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dims", default="256,512,1024")
    ap.add_argument("--hidden-mult", type=int, default=4)
    ap.add_argument("--ks", default="0.5,0.25,0.125")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--B", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"device": "cuda", "error": "no cuda"}))
        return
    if dev.type == "cpu":
        torch.set_num_threads(6)

    res = []
    for d in [int(v) for v in args.dims.split(",")]:
        h = d * args.hidden_mult
        torch.manual_seed(0)
        mk = lambda *s: torch.randn(*s, device=dev) * (1.0 / s[-1] ** 0.5)
        w1 = mk(d, h)
        w2 = mk(h, d)
        b1 = torch.zeros(h, device=dev)
        b2 = torch.zeros(d, device=dev)
        x = torch.randn(args.B, args.T, d, device=dev)

        t_dense = bench(lambda: dense_mlp(x, w1, w2, b1, b2))
        row = {"d": d, "hidden": h, "dense_ms": t_dense}
        for frac in [float(v) for v in args.ks.split(",")]:
            k = max(1, int(h * frac))
            t_s = bench(lambda: sparse_mlp(x, w1, w2, b1, b2, k))
            row[f"topk_k{k}_ms"] = t_s
            row[f"topk_k{k}_speedup"] = t_dense / t_s
        for ne, pt in [(8, 1), (8, 2), (16, 2)]:
            t_m = bench(lambda: moe_mlp(x, w1, w2, b1, b2, ne, pt))
            row[f"moe_e{ne}_k{pt}_ms"] = t_m
            row[f"moe_e{ne}_k{pt}_speedup"] = t_dense / t_m
        for kf in [0.5, 0.25]:
            t_w = bench(lambda: sparse_weight_mm(x, w1, w2, b1, b2, kf))
            row[f"spmm_keep{kf}_ms"] = t_w
            row[f"spmm_keep{kf}_speedup"] = t_dense / t_w
        res.append(row)

    out = {"device": args.device, "B": args.B, "T": args.T, "rows": res}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()