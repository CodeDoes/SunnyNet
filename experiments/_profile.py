"""small 预设单步训练开销剖析 (含反向).

用途: 定位当前架构的算子级瓶颈, 判断优化杠杆.
运行: python -X utf8 experiments/_profile.py [T] [B] [threads] [steps]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--row-limit", type=int, default=22)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    vocab = 8192
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (args.batch, args.seq_len))
    y = torch.randint(0, vocab, (args.batch, args.seq_len))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(args.steps):
            step()

    ev = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in ev)
    print(f"B={args.batch} T={args.seq_len} threads={args.threads} "
          f"steps={args.steps} 自身耗时合计={total/1000:.1f}ms\n")
    print(f"{'op':38s}{'count':>8s}{'self_ms':>10s}{'pct':>7s}")
    for e in sorted(ev, key=lambda e: -e.self_cpu_time_total)[:args.row_limit]:
        pct = 100.0 * e.self_cpu_time_total / max(total, 1e-9)
        print(f"{e.key[:37]:38s}{e.count:>8d}"
              f"{e.self_cpu_time_total/1000:>10.2f}{pct:>6.1f}%")

    print("\n== 分块扫描相关 ==")
    for e in ev:
        if any(k in e.key for k in ("cumsum", "::cat", "slice", "narrow",
                                    "var_mean", "torch.var", "sigmoid")):
            print(f"{e.key[:37]:38s}{e.count:>8d}"
                  f"{e.self_cpu_time_total/1000:>10.2f}")


if __name__ == "__main__":
    main()