"""控制流原语的设备相对成本: 定位稀疏红利的来源.

稀疏在 CPU 上红利更大, 若来自"分支/索引原语相对更便宜", 则这些原语的
CPU/GPU 耗时比应显著低于纯稠密 GEMM 的比值 (13x).

用法: python -X utf8 experiments/_route_mechanism.py --device cpu|cuda
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--d", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(6)

    N, d = args.n, args.d
    x = torch.randn(N, d, device=dev)
    idx = torch.randint(0, d, (N, 8), device=dev)
    mask = torch.rand(N, device=dev) < 0.25
    a = torch.randn(d, d, device=dev) * 0.01
    b = torch.randn(d, d, device=dev) * 0.01

    t = {}
    t["gemm_dxd"] = bench(lambda: a @ b)
    t["gather_cols"] = bench(lambda: x.gather(1, idx))
    t["bool_mask"] = bench(lambda: x[mask])
    t["index_add"] = bench(lambda: torch.zeros(d, device=dev).index_add_(0, idx.reshape(-1) % d, x.reshape(-1)[:idx.numel()]))
    vals, ti = None, None
    t["topk"] = bench(lambda: torch.topk(x, 8, dim=-1))
    t["nonzero"] = bench(lambda: (idx % 2 == 0).nonzero())

    print(json.dumps({"device": args.device, "N": N, "d": d,
                      "ms": {k: round(v, 4) for k, v in t.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()