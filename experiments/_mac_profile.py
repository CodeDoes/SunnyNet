"""各架构单步训练 (前向) 的矩阵乘 MAC 统计, 用 profiler record_shapes 采集.

目的: 判断 SunnyNet 相对同行的速度劣势是"实现低效"还是"结构固有"
(即同参数量下每 token 的稠密乘法量本就更高).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from sunnynet.baseline import TransformerLM, solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import BLOCKS, _LMBase, build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def mac_of(shape):
    """aten::mm(input, other) -> MAC = prod(shape) / last_dim."""
    if len(shape) < 2:
        return 0
    m, k = shape[0], shape[1]
    n = shape[2] if len(shape) > 2 else 1
    return m * k * n


def build(arch, vocab, T):
    ref = SunnyNet(SunnyConfig.small(vocab))
    target = ref.num_params(non_embedding=True)
    del ref
    if arch == "sunnynet":
        return SunnyNet(SunnyConfig.small(vocab))
    if arch == "transformer":
        cfg, _, _ = solve_baseline_for_params(target, vocab, n_layers=6,
                                             n_heads=8, max_len=4096)
        return TransformerLM(cfg)
    cfg, _, _ = build_linear_baseline(arch, target, vocab, max_len=4096,
                                      n_layers=4)
    m = _LMBase(cfg)
    for _ in range(cfg.n_layers):
        m.blocks.append(BLOCKS[arch](cfg))
    m._finish_init()
    return m


def main():
    torch.set_num_threads(8)
    vocab, B, T = 8192, 8, 1024
    for arch in ("sunnynet", "transformer", "linear_attn", "rwkv", "mamba",
                 "hyena", "h3"):
        torch.manual_seed(0)
        m = build(arch, vocab, T).train()
        x = torch.randint(0, vocab, (B, T))
        with torch.no_grad():
            m(x)
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True) as prof:
            with torch.no_grad():
                m(x)
        total_mac = 0
        mm_ops = 0
        for e in prof.events():
            if e.name in ("aten::mm", "aten::addmm", "aten::bmm",
                          "aten::linear", "aten::matmul"):
                for a in e.input_shapes:
                    if len(a) >= 2:
                        total_mac += mac_of(a)
                        break
                mm_ops += 1
        stack = m.num_params(non_embedding=True)
        print(f"{arch:14s} stack={stack:>9d}  mm_ops={mm_ops:>4d}  "
              f"MAC/token={total_mac / (B * T):>10.0f}  "
              f"MAC/token/stackParam={total_mac / (B * T) / stack:>7.3f}")


if __name__ == "__main__":
    main()