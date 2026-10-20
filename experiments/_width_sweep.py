"""参数量对齐下的宽度/层数重平衡: 速度与质量对照.

动机: SunnyNet dim=128 比同行 d_model=88~112 宽, head 与块内 matmul 都更贵.
在栈参数量对齐 (365,844) 前提下枚举 (dim, inner, rank, layers), 找更快的配置.

用法:
  python -X utf8 experiments/_width_sweep.py list
  python -X utf8 experiments/_width_sweep.py bench <tag> <T> <B> <threads>
  python -X utf8 experiments/_width_sweep.py train <tag> <steps> <T> <B> <seed>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

TARGET = 365844


def stack_params(dim, inner, rank, layers):
    """meta device 实例化计数, 不手推公式 (漏项会到 30% 量级)."""
    cfg = SunnyConfig(vocab_size=8, dim=dim, inner=inner, rank=rank,
                      layers=layers, h_max=2048.0)
    with torch.device("meta"):
        m = SunnyNet(cfg)
    return m.num_params(non_embedding=True)


def configs():
    out = {}
    for dim in (80, 88, 96, 104, 112, 120, 128, 144):
        for layers in (4, 5, 6, 7):
            best = None
            for rank in range(32, min(dim, 160) + 1, 8):
                for inner in range(64, dim * 3 + 1, 8):
                    n = stack_params(dim, inner, rank, layers)
                    if n > TARGET * 1.35:
                        break
                    rel = abs(n - TARGET) / TARGET
                    if best is None or rel < best[0]:
                        best = (rel, n, inner, rank)
            if best and best[0] < 0.03:
                rel, n, inner, rank = best
                tag = f"d{dim}_i{inner}_r{rank}_L{layers}"
                out[tag] = dict(dim=dim, inner=inner, rank=rank, layers=layers,
                                stack=n, rel=rel)
    return out


CFG = configs()


def build(tag, vocab):
    c = CFG[tag]
    return SunnyNet(SunnyConfig(vocab_size=vocab, dim=c["dim"], inner=c["inner"],
                                rank=c["rank"], layers=c["layers"], h_max=2048.0))


def bench(fn, reps=11, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000, ts[0] * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "bench", "train"])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("a", nargs="?", type=int, default=1024)
    ap.add_argument("b", nargs="?", type=int, default=8)
    ap.add_argument("c", nargs="?", type=int, default=8)
    ap.add_argument("d", nargs="?", type=int, default=1234)
    args = ap.parse_args()

    if args.cmd == "list":
        print(f"{'tag':26s}{'stack':>10s}{'rel':>8s}  dim/inner/rank/L")
        for t, c in sorted(CFG.items(), key=lambda kv: kv[1]["stack"]):
            print(f"{t:26s}{c['stack']:>10,}{c['rel']*100:>7.2f}%  "
                  f"{c['dim']}/{c['inner']}/{c['rank']}/{c['layers']}")
        print(f"\n目标栈参数 {TARGET:,}; 共 {len(CFG)} 个候选")
        return

    torch.set_num_threads(args.c)
    vocab = 8192
    tag = args.tag or "d128_i192_r64_L4"
    if tag == "base":
        tag = "d128_i192_r64_L4"
    torch.manual_seed(0)
    m = build(tag, vocab)

    if args.cmd == "bench":
        T, B = args.a, args.b
        x = torch.randint(0, vocab, (B, T))
        y = torch.randint(0, vocab, (B, T))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

        def step():
            lg = m(x)
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        step()
        ms, mn = bench(step, reps=11, warm=3)
        print("RESULT " + json.dumps({
            "tag": tag, "stack": m.num_params(non_embedding=True),
            "step_ms": round(ms, 2), "min_ms": round(mn, 2),
            "T": T, "B": B, "threads": args.c,
        }))
        return

    # train: 质量对照
    steps, T, B, seed = args.a, args.b, args.c, args.d
    torch.manual_seed(seed)
    m = build(tag, vocab).train()
    data = torch.randint(0, vocab, (B, T * 64))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    best = float("inf")
    for it in range(steps):
        i = (it * T) % (data.shape[1] - T - 1)
        xb = data[:, i:i + T]
        yb = data[:, i + 1:i + T + 1]
        lg = m(xb)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        i = 0
        xb = data[:, i:i + T]
        yb = data[:, i + 1:i + T + 1]
        lg = m(xb)
        val = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), yb.reshape(-1)).item()
    print("RESULT " + json.dumps({
        "tag": tag, "steps": steps, "seed": seed,
        "val_loss": round(val, 5), "ppl": round(math.exp(val), 2),
        "stack": m.num_params(non_embedding=True),
    }))


if __name__ == "__main__":
    main()