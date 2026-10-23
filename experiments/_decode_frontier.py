"""B=1 长上下文延迟前沿: CPU 真正取胜区间的边界.

递推架构的单步延迟不随 T 增长 (O(1)), Transformer 随 T 线性 (KV cache 读取).
CPU 侧叠加 GPU 的固定启动地板, 故存在一个 T* 使 CPU-SunnyNet 追平 GPU-Transformer.

本实验扫 T, 记录单步解码延迟, 给出交点与斜率.

用法: python -X utf8 experiments/_decode_frontier.py --device cpu|cuda
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

from sunnynet.baseline import TransformerLM, solve_baseline_for_params  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

VOCAB = 8192


def bench_step(fn, warmup=5, iters=30, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--Ts", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    if not cuda:
        torch.set_num_threads(6)

    cfg = SunnyConfig(vocab_size=VOCAB, dim=256, inner=384, rank=128, layers=4,
                      h_max=2048.0)
    cfg.scan_chunk = 64
    sunny = SunnyNet(cfg).to(dev).eval()
    target = sunny.num_params(non_embedding=True)
    bcfg, bn, _ = solve_baseline_for_params(target, VOCAB, n_layers=4, n_heads=8,
                                            max_len=16384)
    tf = TransformerLM(bcfg).to(dev).eval()

    rows = []
    for T in [int(v) for v in args.Ts.split(",")]:
        idx = torch.randint(0, VOCAB, (1, T), device=dev)
        with torch.no_grad():
            # 两边都用单步路径 (KV cache 已建), 才是同口径
            _, st = sunny(idx[:, :T - 1], return_state=True)
            one = idx[:, 0]
            t_sn = bench_step(lambda: sunny.forward_step(one, st), cuda=cuda,
                              iters=args.iters)
            # Transformer: 逐步建 KV cache 到长度 T, 再测单步
            tstate = None
            for t in range(T):
                _, tstate = tf.forward_step(idx[:, t], tstate)
            t_tf = bench_step(lambda: tf.forward_step(one, tstate), cuda=cuda,
                              iters=args.iters)
        rows.append({"T": T, "sunny_ms": round(t_sn, 4), "tf_ms": round(t_tf, 4),
                     "ratio_tf_over_sn": round(t_tf / t_sn, 3)})

    print(json.dumps({"device": args.device, "params": bn, "rows": rows},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()