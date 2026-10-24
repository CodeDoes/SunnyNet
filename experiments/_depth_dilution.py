"""深度阶梯: 层间输入特异性随深度的稀释曲线, 以及 h_norm 的因果效果.

判据: 逐层残差流的"输入特异性能量占比" = 1 - ||跨输入公共均值||^2 / mean||h||^2.
该占比随深度单调下降 = 深度压缩; h_norm 打开后是否把它拉回.

同时给出每个配置的 val ppl 与输出多样度 (argmax 去重数), 判断稀释是否伴随质量损失.

用法: python -X utf8 experiments/_depth_dilution.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB = 8192


def load(domain, split="train"):
    return np.memmap(ROOT / f"data/proc/{'tokens' if split == 'train' else 'val'}_{domain}.bin",
                     dtype=np.uint16, mode="r")


def spec_curve(model, xs, cfg):
    """逐层输入特异性能量占比, 用真实块前向 (状态从零起)."""
    L, D = cfg.layers, cfg.dim
    k = max(0, cfg.taps - 1)
    from sunnynet.ops import logspace_scan
    ratios, abs_common, abs_spec = [], [], []
    with torch.no_grad():
        h = model.embed(xs)
        s = torch.zeros(xs.shape[0], L, D)
        hist = torch.zeros(xs.shape[0], L, k, D)
        mem = None
        a0 = next((b.assoc for b in model.blocks if b.assoc is not None), None)
        if a0 is not None:
            mem = torch.zeros(xs.shape[0], L, a0.slots, a0.m)
        for li, blk in enumerate(model.blocks):
            if blk.assoc is not None:
                h, s_i, h_i, m_i = blk(h, s[:, li], hist[:, li], mem[:, li])
                mem[:, li] = m_i
            else:
                h, s_i, h_i = blk(h, s[:, li], hist[:, li])
            if model.h_norm is not None:
                h = model.h_norm(h)
            s[:, li] = s_i
            hist[:, li] = h_i
            hmu = h.mean(dim=1)                        # (n, D) 逐样本时间均值
            common = hmu.mean(0)
            spec = hmu - common
            e_all = hmu.norm(dim=-1).pow(2).mean()
            ratios.append(round(float(1.0 - common.norm().pow(2) / e_all.clamp_min(1e-12)), 4))
            nrm = h.norm(dim=1).pow(2).mean().sqrt()
            abs_common.append(round(float(common.norm() / nrm.clamp_min(1e-9)), 4))
            abs_spec.append(round(float(spec.norm(dim=-1).pow(2).mean().sqrt()
                                       / nrm.clamp_min(1e-9)), 4))
    return ratios, abs_common, abs_spec


def measure(model, tr, T, n_in):
    starts = torch.randint(0, len(tr) - T - 1, (n_in,))
    xs = torch.stack([torch.from_numpy(np.asarray(tr[s:s + T], dtype=np.int64))
                      for s in starts])
    with torch.no_grad():
        logits = model(xs)
        tl = logits[:, -1]
        am = int(tl.argmax(-1).unique().numel())
        lp = torch.log_softmax(tl, dim=-1)
        ent = float((-(lp.exp() * lp).sum(-1)).mean())
    return xs, am, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--ablation", default="")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n-in", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from train.train import _preset_cfg, lr_at
    from sunnynet.model import SunnyNet

    ablation = {}
    if args.ablation:
        for kv in args.ablation.split(","):
            k, v = kv.split("=")
            try:
                iv = int(v)
            except ValueError:
                ablation[k] = v
            else:
                ablation[k] = bool(iv) if iv in (0, 1) else iv

    tr = load("zh_ancient")
    cfg = _preset_cfg(args.preset, VOCAB)
    cfg.scan_chunk = 64
    if ablation:
        cfg.ablation = dict(ablation)
    torch.manual_seed(1234)
    model = SunnyNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, args.steps // 20)

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warm)
        i = int(torch.randint(0, len(tr) - args.B * args.T - 2, (1,)))
        buf = np.asarray(tr[i:i + args.B * args.T + 1], dtype=np.int64)
        x = torch.from_numpy(buf[:-1]).reshape(args.B, args.T)
        y = torch.from_numpy(buf[1:]).reshape(args.B, args.T)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # 验证 ppl
    va = load("zh_ancient", "val")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(16):
            i = int(torch.randint(0, len(va) - args.T - 2, (1,)))
            w = torch.from_numpy(np.asarray(va[i:i + args.T + 1], dtype=np.int64))
            lg = model(w[None, :-1])
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, VOCAB), w[1:], reduction="sum"))
            n += args.T
    import math
    ppl = round(math.exp(min(tot / n, 20)), 2)

    xs, am, ent = measure(model, tr, args.T, args.n_in)
    ratios, ac, aspec = spec_curve(model, xs, cfg)
    print(json.dumps({
        "tag": args.tag, "preset": args.preset, "ablation": args.ablation,
        "layers": cfg.layers, "dim": cfg.dim, "steps": args.steps,
        "stack_params": model.num_params(non_embedding=True),
        "val_ppl_zh": ppl,
        "spec_ratio_by_layer": ratios,
        "abs_common_rms_by_layer": ac,
        "abs_spec_rms_by_layer": aspec,
        "first_to_last_spec_drop": round(ratios[0] - ratios[-1], 4),
        "argmax_unique": am, "n_in": args.n_in,
        "last_logit_entropy_nats": round(ent, 4),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()