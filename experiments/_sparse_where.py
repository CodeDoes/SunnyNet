"""稀疏红利的落点: 在哪个场景能兑现?

已知: 训练端到端 MoE 红利为 0.884x (未穿透).
但稀疏红利来自"多而小 GEMM 的 CPU 相对效率", 这在不同场景兑现程度不同:
 - B=1 解码: 每步仅 1 token, 专家路由无意义 (只激活 1 个) -> 预期无红利;
 - 长 prompt 预填充 (prefill): 多 token 同时过, 专家可分批 -> 预期有红利;
 - 训练: 反向物化更多中间量 -> 已知不穿透.

本实验区分这三种, 给出稀疏的正确落点.

用法: python -X utf8 experiments/_sparse_where.py --device cpu|cuda
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


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def dense_fwd(x, w1, w2):
    return F.gelu(x @ w1) @ w2


def moe_fwd(x, w1, w2, E):
    """E 专家, 每 token 走 1 个. 无反向, 纯前向."""
    N, d = x.shape
    h = w1.shape[1] // E
    topi = torch.arange(N, device=x.device) % E
    out = torch.zeros(N, d, device=x.device)
    for e in range(E):
        rows = (topi == e).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        out.index_add_(0, rows, F.gelu(x[rows] @ w1[:, e * h:(e + 1) * h]) @ w2[e * h:(e + 1) * h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--hidden-mult", type=int, default=4)
    ap.add_argument("--experts", default="4,8,16")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)
    d = args.d
    h = d * args.hidden_mult
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.01).to(dev)
    w1, w2 = mk(d, h), mk(h, d)

    rows = []
    scenarios = [("decode_b1", 1), ("prefill_1k", 1024), ("prefill_8k", 8192),
                 ("train_b8t256", 8 * 256)]
    for name, N in scenarios:
        x = (torch.randn(N, d, device=dev) * 0.02)
        t_d = bench(lambda: dense_fwd(x, w1, w2))
        row = {"scenario": name, "tokens": N, "dense_ms": round(t_d, 4)}
        for E in [int(v) for v in args.experts.split(",")]:
            t_m = bench(lambda: moe_fwd(x, w1, w2, E))
            row[f"E{E}_dividend"] = round(t_d / t_m, 3)
        rows.append(row)

    print(json.dumps({"device": args.device, "d": d, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()