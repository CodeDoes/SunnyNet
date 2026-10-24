"""状态坍缩探针: 判定"不连贯"是否来自输入被状态反馈抹平.

机制假设: 块内状态反馈 s_t = p*s_{t-1} + (1-p)*u_t 之后再叠加 W_s*circ(s),
若反馈回路把输入信息压掉, 不同输入会收敛到同一状态 -> 生成退化为固定句式.

判据:
 1) 不同输入在末层状态的成对余弦相似度 -> 接近 1 = 坍缩;
 2) 打乱输入状态的顺序, 看状态变化的相对幅度 (敏感度);
 3) 逐层看相似度如何累积 (定位在哪一层坍缩).

用法: python -X utf8 experiments/_state_collapse.py --preset small --steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB = 8192


def load(domain):
    tr = np.memmap(ROOT / f"data/proc/tokens_{domain}.bin", dtype=np.uint16, mode="r")
    va = np.memmap(ROOT / f"data/proc/val_{domain}.bin", dtype=np.uint16, mode="r")
    return (torch.from_numpy(np.asarray(tr, dtype=np.int64)),
            torch.from_numpy(np.asarray(va, dtype=np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--domain", default="zh_ancient")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n-in", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ablation", default="",
                    help='如 "assoc=1,assoc_slots=32,assoc_m=16"')
    args = ap.parse_args()

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

    torch.set_num_threads(6)
    from train.train import _preset_cfg, build_models, lr_at
    from sunnynet.model import SunnyNet

    tr, _ = load(args.domain)
    cfg = _preset_cfg(args.preset, VOCAB)
    cfg.scan_chunk = 64
    if ablation:
        cfg.ablation = dict(ablation)
    model = SunnyNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, max(1, args.steps // 20))
        i = int(torch.randint(0, len(tr) - args.B * args.T - 2, (1,)))
        x = tr[i:i + args.B * args.T].reshape(args.B, args.T)
        y = tr[i + 1:i + 1 + args.B * args.T].reshape(args.B, args.T)
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    # 取 n_in 个互不相同的输入片段
    T = args.T
    starts = torch.randint(0, len(tr) - T - 1, (args.n_in,))
    xs = torch.stack([tr[s:s + T] for s in starts])

    with torch.no_grad():
        # 逐层状态 + 残差流: 复用块接口收集
        per_layer = []
        h_layer = [model.embed(xs).mean(dim=1)]
        s = torch.zeros(args.n_in, cfg.layers, cfg.dim)
        hist = torch.zeros(args.n_in, cfg.layers, max(0, cfg.taps - 1), cfg.dim)
        a0 = next((b.assoc for b in model.blocks if b.assoc is not None), None)
        mem = (torch.zeros(args.n_in, cfg.layers, a0.slots, a0.m)
               if a0 is not None else None)
        h = model.embed(xs)
        for li, blk in enumerate(model.blocks):
            if blk.assoc is not None:
                h, s_i, h_i, m_i = blk(h, s[:, li], hist[:, li], mem[:, li])
                mem[:, li] = m_i
            else:
                h, s_i, h_i = blk(h, s[:, li], hist[:, li])
            s[:, li] = s_i
            hist[:, li] = h_i
            per_layer.append(s[:, li].clone())          # (n_in, dim)
            h_layer.append(h.mean(dim=1).clone())        # 残差流时间平均 (n_in, dim)
        last_state = per_layer[-1]
        trained_logits = model.head(model.norm_out(h))   # (n_in, T, V)
        tl = trained_logits[:, -1]

    def pair_cos(mat):
        v = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        c = v @ v.t()
        n = c.shape[0]
        iu = torch.triu_indices(n, n, offset=1)
        return float(c[iu[0], iu[1]].mean())

    layer_cos = [round(pair_cos(p), 4) for p in per_layer]
    h_cos = [round(pair_cos(p), 4) for p in h_layer]
    sc = pair_cos(last_state)
    tl_cos = pair_cos(tl)
    # 逐 token 位置的平均成对余弦 (残差流), 判断坍缩是逐 token 全局还是仅在末端
    with torch.no_grad():
        hh = model.embed(xs)
        h_all = []
        for li, blk in enumerate(model.blocks):
            if blk.assoc is not None:
                hh, _, _, _ = blk(hh, torch.zeros_like(s[:, li]),
                                  torch.zeros_like(hist[:, li]),
                                  torch.zeros_like(mem[:, li]))
            else:
                hh, _, _ = blk(hh, torch.zeros_like(s[:, li]),
                               torch.zeros_like(hist[:, li]))
            h_all.append(hh.clone())
        h_pos_cos = [round(pair_cos(h_all[-1][:, p]), 4)
                     for p in (0, args.T // 2, args.T - 1)]

    # 敏感度: 两个不同输入的状态相对差
    diff = (last_state[0] - last_state[1]).norm() / last_state[0].norm().clamp_min(1e-9)

    # 未训练基线 (随机初始化) 作对照
    torch.manual_seed(999)
    m0 = SunnyNet(cfg).eval()
    with torch.no_grad():
        logits0 = m0(xs)
        l0 = logits0[:, -1]
    v0 = l0 / l0.norm(dim=-1, keepdim=True)
    c0 = v0 @ v0.t()
    iu = torch.triu_indices(args.n_in, args.n_in, offset=1)
    rand_cos = float(c0[iu[0], iu[1]].mean())

    print(json.dumps({
        "preset": args.preset, "domain": args.domain, "steps": args.steps,
        "layer_pair_cos": layer_cos,
        "residual_layer_cos": h_cos,
        "residual_pos_cos_last_layer": h_pos_cos,
        "last_state_pair_cos": round(sc, 4),
        "trained_logit_cos": round(tl_cos, 4),
        "state_rel_diff_two_inputs": round(float(diff), 4),
        "random_init_logit_cos": round(rand_cos, 4),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()