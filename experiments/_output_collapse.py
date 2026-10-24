"""输出坍缩探针: 从收敛 checkpoint 测"不连贯"的量化根因.

"不连贯"的机制假设不是逐 token 噪声, 而是模型对任意输入给出几乎相同的
下一 token 分布 —— 输出去条件化. 判据:

 1) 不同输入片段 (末位) logit 的成对余弦 -> 接近 1 = 输出坍缩;
 2) 相邻 token 位置的 logit 余弦 (同一条序列内);
 3) 与随机初始化模型的同口径对照 (排除"logit 天然相似"的解释).

用法: python -X utf8 experiments/_output_collapse.py --ckpt runs/base_small_zh.pt
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


def pair_cos(mat):
    v = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = v @ v.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(c[iu[0], iu[1]].mean())


def pair_js(logits):
    """成对 JS 散度 (nats): 分布层面的趋同度, 不受公共分量误导."""
    p = torch.softmax(logits, dim=-1)
    n = p.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    a, b = p[iu[0]], p[iu[1]]
    m = 0.5 * (a + b)
    def kl(x, y):
        return (x * (x.clamp_min(1e-12).log() - y.clamp_min(1e-12).log())).sum(-1)
    js = 0.5 * kl(a, m) + 0.5 * kl(b, m)
    return float(js.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain", default="zh_ancient")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n-in", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(6)

    from train.evaluate import load_model
    model, cfg = load_model(args.ckpt)

    tr = np.memmap(ROOT / f"data/proc/tokens_{args.domain}.bin",
                   dtype=np.uint16, mode="r")
    starts = torch.randint(0, len(tr) - args.T - 1, (args.n_in,))
    xs = torch.stack([torch.from_numpy(np.asarray(tr[s:s + args.T], dtype=np.int64))
                      for s in starts])

    with torch.no_grad():
        logits = model(xs)                       # (n_in, T, V)

    # 1) 末位 logit 成对余弦
    last_cos = pair_cos(logits[:, -1])
    # 1b) 去均值后的成对余弦: 排除"共享一个巨大公共分量"的假坍缩
    centered = logits[:, -1] - logits[:, -1].mean(0, keepdim=True)
    centered_cos = pair_cos(centered)
    # 2) 同一序列内, 位置 p 处不同输入之间的 logit 余弦 (逐位置)
    pos_cos = [round(pair_cos(logits[:, p]), 4)
               for p in (0, args.T // 4, args.T // 2, args.T - 1)]
    # 2b) 公共分量能量占比: ||mean(logit)|| / RMS(||logit||), 越大说明输出越被输入无关项主导
    mu = logits[:, -1].mean(0)
    pub_ratio = float(mu.norm() / logits[:, -1].norm(dim=-1).pow(2).mean().sqrt())
    # 3) 输出分布的熵 (取值越集中 -> 越"固定句式")
    with torch.no_grad():
        lp = torch.log_softmax(logits[:, -1], dim=-1)
        ent = float((-(lp.exp() * lp).sum(-1)).mean())
        # 每输入 top-1 token 的一致性: 32 个输入里有多少给出同一个 argmax
        am = logits[:, -1].argmax(-1)
        uniq = int(am.unique().numel())

    # 随机初始化对照 (同架构重建)
    torch.manual_seed(999)
    if hasattr(cfg, "ablation"):
        from sunnynet.model import SunnyNet
        m0 = SunnyNet(cfg).eval()
    else:
        from sunnynet.baseline import BaselineConfig, TransformerLM
        m0 = TransformerLM(BaselineConfig(**cfg.__dict__)).eval()
    with torch.no_grad():
        l0 = m0(xs)[:, -1]
    rand_cos = pair_cos(l0)

    print(json.dumps({
        "ckpt": args.ckpt, "domain": args.domain, "T": args.T, "n_in": args.n_in,
        "last_logit_pair_cos": round(last_cos, 4),
        "last_logit_centered_cos": round(centered_cos, 4),
        "last_logit_pair_js_nats": round(pair_js(logits[:, -1]), 4),
        "public_component_ratio": round(pub_ratio, 4),
        "pos_logit_pair_cos": pos_cos,
        "last_logit_entropy_nats": round(ent, 4),
        "argmax_unique_in_32": uniq,
        "random_init_pair_cos": round(rand_cos, 4),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()