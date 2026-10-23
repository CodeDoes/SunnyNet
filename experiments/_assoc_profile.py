"""剖析关联记忆的耗时构成, 找 CPU 上的瓶颈.

关联记忆把每层状态从 D 个标量升到 H*m*m 个 (H*m*m = D^2/H),
读出与扫描都变成 O(D^2) 量级. 本脚本逐段计时, 定位该压哪一块.

用法: python -X utf8 experiments/_assoc_profile.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet


def bench(fn, iters=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(6)

    def cfg_of(ab):
        return SunnyConfig(vocab_size=args.vocab, dim=args.dim, inner=args.dim * 2,
                           rank=args.dim // 2, layers=args.layers, ablation=ab)

    torch.manual_seed(0)
    m0 = SunnyNet(cfg_of({}))
    torch.manual_seed(0)
    m1 = SunnyNet(cfg_of({"assoc": True, "assoc_heads": 4}))
    idx = torch.randint(0, args.vocab, (args.B, args.T))
    y = torch.randint(0, args.vocab, (args.B, args.T))

    def fwd_bwd(m):
        loss = torch.nn.functional.cross_entropy(m(idx).reshape(-1, args.vocab),
                                                 y.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
        return loss

    t_base = bench(lambda: fwd_bwd(m0))
    t_assoc = bench(lambda: fwd_bwd(m1))

    blk = m1.blocks[0]
    a = blk.norm_in(m1.embed(idx))
    mem = torch.zeros(args.B, blk.assoc.heads, blk.assoc.m, blk.assoc.m)
    t_assoc_fwd = bench(lambda: blk.assoc(a, mem))
    xa = a.detach().requires_grad_(True)
    memg = mem.detach().requires_grad_(True)

    def ab():
        o, _ = blk.assoc(xa, memg)
        o.sum().backward()
        xa.grad = None
        memg.grad = None

    t_assoc_fb = bench(ab, iters=10)

    print(f"dim={args.dim} layers={args.layers} B={args.B} T={args.T}")
    print(f"  全模型 fwd+bwd  基线  : {t_base:8.2f} ms")
    print(f"  全模型 fwd+bwd  +关联 : {t_assoc:8.2f} ms   ({t_assoc/t_base:.2f}x)")
    print(f"  单块 assoc 前向       : {t_assoc_fwd:8.2f} ms")
    print(f"  单块 assoc 前向+反向  : {t_assoc_fb:8.2f} ms")
    print(f"  状态维度: D={args.dim} -> H*m*m = "
          f"{blk.assoc.heads * blk.assoc.m * blk.assoc.m}  ({blk.assoc.heads * blk.assoc.m * blk.assoc.m / args.dim:.0f}x)")


if __name__ == "__main__":
    main()