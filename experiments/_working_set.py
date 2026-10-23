"""工作集驻留: 现有预设的权重 + 激活 + 梯度 + 优化器态字节数, 对照缓存容量.

CPU 高效区由 §3 的缓存断崖界定 (L2 1.5MB / L3 9MB, 实用高带宽区 ~3-4MB)。
本脚本按真实训练步统计每类张量的字节数, 给出现有预设落在哪一档。

口径: 只统计张量本体字节 (fp32=4B/元素), 不含框架开销。
激活按 B/T 实算 —— 这是训练步内存的主项, 也是唯一随 B/T 变的部分。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from train import build_models  # noqa: E402

L2_KB, L3_KB = 1536, 9216
EFF_KB = 4096          # §3 实测: 4096KB 起离开高带宽区


def mb(n_bytes: int) -> float:
    return n_bytes / 1024 / 1024


def classify(model):
    """权重按用途分三类, 决定它们能否驻留缓存."""
    tot = {}
    for name, p in model.named_parameters():
        n = p.numel() * p.element_size()
        if "embed" in name or "head" in name or "tok" in name:
            tot["embed/head"] = tot.get("embed/head", 0) + n
        elif p.dim() >= 2:
            tot["matrix"] = tot.get("matrix", 0) + n
        else:
            tot["vector/scalar"] = tot.get("vector/scalar", 0) + n
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    sunny, _, cfg = build_models("sunnynet", args.vocab, args.preset, None,
                                 args.seq_len, 1234)
    d = cfg.dim
    L = cfg.layers
    B, T = args.batch_size, args.seq_len

    par = classify(sunny)
    w_bytes = sum(par.values())

    # 训练步常驻: 参数 + 梯度 + AdamW 一阶/二阶矩 (各一份 fp32)
    train_state = w_bytes * 4

    # 激活上界 (单层峰值, 按最宽张量估): norm_mix(inner) + q(2d) + states(T*d) + ...
    inner = cfg.inner
    act_layer = B * T * (inner * 4 + d * 3) * 4      # 前向若干最宽张量的粗估
    act_total = act_layer * L

    print(f"preset={args.preset}  d={d} inner={inner} L={L}  B={B} T={T}")
    print(f"  参数总量        {mb(w_bytes):>8.2f} MB")
    for k, v in sorted(par.items()):
        print(f"    {k:<14} {mb(v):>8.2f} MB")
    print(f"  参数+梯度+AdamW {mb(train_state):>8.2f} MB")
    print(f"  激活(单层估)    {mb(act_layer):>8.2f} MB")
    print(f"  激活(全层估)    {mb(act_total):>8.2f} MB")
    print()
    print(f"  缓存: L2 {L2_KB/1024:.1f} MB | L3 {L3_KB/1024:.1f} MB | 高效区上限 {EFF_KB/1024:.1f} MB")
    print()
    # 判定: 哪些类别能驻留
    for k, v in sorted(par.items()):
        tag = "L2内" if v <= L2_KB * 1024 else ("L3内" if v <= L3_KB * 1024 else "溢出L3")
        print(f"  [{k:<14}] {mb(v):>7.2f} MB -> {tag}")
    single_layer = act_layer + w_bytes / L
    print(f"\n  单层激活+分摊权重 {mb(single_layer):.2f} MB -> "
          f"{'在高效区' if single_layer <= EFF_KB*1024 else '超出高效区'}")


if __name__ == "__main__":
    main()