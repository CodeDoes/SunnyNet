"""块内宽度 vs 训练速度: 同参数量下重分配 dim/inner/rank/layers 是否更快.

SunnyNet small: dim=128 inner=192 rank=64 layers=4  (每层最窄)
同行在等参数下 d_model 88-112 但内层更宽.

本脚本在**参数量对齐**前提下扫几种宽度分配, 测训练步耗时.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def bench(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    B = 8
    torch.set_num_threads(8)
    torch.manual_seed(0)
    vocab = 8192

    # 目标栈参数 = 当前 small
    ref = SunnyNet(SunnyConfig.small(vocab))
    target = ref.num_params(non_embedding=True)
    del ref

    if mode == "base":
        cfg = SunnyConfig.small(vocab)
    elif mode == "wide":
        # 更宽更浅: 层数减到 3, 靠 dim/inner 补回参数
        cfg = SunnyConfig(vocab_size=vocab, dim=192, inner=288, rank=96,
                          layers=3)
    elif mode == "wider":
        cfg = SunnyConfig(vocab_size=vocab, dim=224, inner=320, rank=112,
                          layers=2)
    elif mode == "deep":
        cfg = SunnyConfig(vocab_size=vocab, dim=96, inner=160, rank=52,
                          layers=7)
    else:
        raise SystemExit(f"unknown mode {mode}")

    m = SunnyNet(cfg).train()
    nst = m.num_params(non_embedding=True)
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    ms = bench(step, reps=7, warm=2)
    print(f"RESULT mode={mode} T={T} dim={cfg.dim} inner={cfg.inner} "
          f"rank={cfg.rank} layers={cfg.layers} stack={nst} "
          f"dev={100*(nst-target)/target:+.1f}% step_ms={ms:.1f}")


if __name__ == "__main__":
    main()