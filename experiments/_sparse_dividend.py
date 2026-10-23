"""稀疏红利随规模的变化: 同一份稀疏实现, CPU 与 GPU 各获益多少.

定义 稀疏红利 = 稠密耗时 / 稀疏耗时 (等有效 FLOPs 下).
若 CPU 的红利随分支数与 token 数上升, 而 GPU 的红利被启动开销吞掉,
则"数据相关控制流"是 CPU 的结构优势 —— 这是本项目要找的 CPU 出路之一.

诚实边界: GPU 侧用 eager 逐专家循环, 生产级 MoE 会用融合分组 GEMM 缩小差距.
本实验测量的是"两者在通用 PyTorch 下"的相对位置, 不声称 GPU 无法优化.

用法: python -X utf8 experiments/_sparse_dividend.py --device cpu|cuda
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


def bench(fn, warmup=3, iters=15):
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


def make(d, h, dev):
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s: (torch.randn(*s, generator=g) * (1.0 / s[-1]) ** 0.5).to(dev)
    return r(d, h), r(h, d), torch.zeros(h, device=dev), torch.zeros(d, device=dev)


def dense(x, w1, w2, b1, b2):
    return torch.relu(x @ w1 + b1) @ w2 + b2


def moe(x, w1, w2, b1, b2, n_expert):
    """E 个等大专家, 每 token 只走 1 个: 有效 FLOPs = 稠密 / E."""
    B, S, d = x.shape
    h = w1.shape[1] // n_expert
    N = B * S
    fx = x.reshape(N, d)
    gate = torch.softmax(fx @ w1[:, :n_expert] * 0.0 + 1.0, dim=-1)  # 均匀路由
    topi = (torch.arange(N, device=x.device) % n_expert)
    out = torch.zeros(N, d, device=x.device)
    for e in range(n_expert):
        rows = (topi == e).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        xe = fx[rows]
        he = torch.relu(xe @ w1[:, e * h:(e + 1) * h] + b1[e * h:(e + 1) * h])
        out.index_add_(0, rows, he @ w2[e * h:(e + 1) * h] + b2)
    return out.reshape(B, S, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--hidden-mult", type=int, default=4)
    ap.add_argument("--experts", default="2,4,8,16,32")
    ap.add_argument("--tokens", default="512,2048,8192")
    ap.add_argument("--B", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)
    d = args.d
    h = d * args.hidden_mult
    w1, w2, b1, b2 = make(d, h, dev)

    rows = []
    for ntok in [int(v) for v in args.tokens.split(",")]:
        S = max(1, ntok // args.B)
        x = torch.randn(args.B, S, d, device=dev) * 0.02
        t_dense = bench(lambda: dense(x, w1, w2, b1, b2))
        row = {"tokens": args.B * S, "dense_ms": round(t_dense, 4)}
        for E in [int(v) for v in args.experts.split(",")]:
            t_m = bench(lambda: moe(x, w1, w2, b1, b2, E))
            row[f"E{E}_ms"] = round(t_m, 4)
            row[f"E{E}_dividend"] = round(t_dense / t_m, 3)
        rows.append(row)

    print(json.dumps({"device": args.device, "d": d, "hidden": h,
                      "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()