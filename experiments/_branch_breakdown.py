"""块内分支分解探针: 定位"输入特异性"在哪一支被压掉.

每层 out = W_c(y) + b_gain*sn + sfb + x:
  y   由当前输入经 readout 得来, 与累积状态无关;
  sn  状态读出 (norm_s(states));
  sfb 状态反馈 W_s(circ(prev));
  x   残差输入.

对每一支分别测**跨输入的成对余弦**(原始 / 去均值)与范数占比:
  原始余弦高      = 该支在不同输入间趋同;
  去均值后仍高    = 真坍缩 (不是公共分量);
  范数占比高      = 该支主导输出.

用法: python -X utf8 experiments/_branch_breakdown.py --ckpt runs/base_small_zh.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pair_cos(mat):
    v = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = v @ v.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return round(float(c[iu[0], iu[1]].mean()), 4)


def centered_cos(mat):
    return pair_cos(mat - mat.mean(0, keepdim=True))


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
    L, D = cfg.layers, cfg.dim

    tr = np.memmap(ROOT / f"data/proc/tokens_{args.domain}.bin",
                   dtype=np.uint16, mode="r")
    starts = torch.randint(0, len(tr) - args.T - 1, (args.n_in,))
    xs = torch.stack([torch.from_numpy(np.asarray(tr[s:s + args.T], dtype=np.int64))
                      for s in starts])

    k = max(0, cfg.taps - 1)
    rows = []
    with torch.no_grad():
        h = model.embed(xs)
        s = torch.zeros(args.n_in, L, D)
        hist = torch.zeros(args.n_in, L, k, D)
        for li, blk in enumerate(model.blocks):
            if blk.assoc is not None:
                break
            raw = blk.norm_in(h)
            a = blk._apply_local(raw, hist[:, li])
            u, gate, p = blk._core(a)
            from sunnynet.ops import logspace_scan
            states, _ = logspace_scan(u, p, s[:, li], chunk=blk.scan_chunk,
                                      p_min_hint=blk.p_min_bound)
            y = blk._readout(u, None)
            sn = blk.norm_s(states) if blk.norm_s is not None else states
            prev = blk._prev(states, s[:, li])
            sfb = (F.linear(prev, blk._Ws_eff()) if blk.circ is not None
                   else blk.W_s(prev))
            part_wc = blk.W_c(y)
            part_sn = blk.b_gain * sn
            part_sfb = sfb
            part_x = h
            core = part_wc + part_sn + part_sfb
            out = core + part_x

            if k:
                hist[:, li] = raw[:, args.T - k:args.T]
            s[:, li] = states[:, -1]
            h = out

            # 时间平均到 (n_in, D), 跨输入比较
            def tm(t):
                return t.mean(dim=1)
            on = tm(out).norm(dim=-1).mean()

            br = {}
            for name, t in (("y", part_wc), ("sn", part_sn), ("sfb", part_sfb),
                            ("x", part_x), ("core", core), ("out", out)):
                mt = tm(t)
                br[name] = {
                    "cos": pair_cos(mt),
                    "ccos": centered_cos(mt),
                    "norm_frac": round(float(mt.norm(dim=-1).mean() / on.clamp_min(1e-9)), 4),
                }

            # DC 是时间慢漂移还是纯跨输入共享? 逐样本时间均值 (B,D) 的能量占比,
            # 以及这些时间均值的跨样本余弦. 占比高 = 单个样本内部就有一个大 DC.
            def temporal_stats(t):
                tmean = t.mean(dim=1)                       # (n_in, D)
                e_t = tmean.norm(dim=-1).pow(2).mean()
                e_all = t.norm(dim=-1).pow(2).mean()
                return (round(float(e_t / e_all.clamp_min(1e-12)), 4), pair_cos(tmean))
            out_t_ratio, out_t_cos = temporal_stats(out)
            # 预言机高通 (逐样本减去自身时间均值, 非因果, 作上界):
            out_hp = out - out.mean(dim=1, keepdim=True)
            hp_pub = round(float(out_hp.mean(0).norm().pow(2)
                                 / out_hp.norm(dim=-1).pow(2).mean().clamp_min(1e-12)), 4)
            # 逐层残差流的公共/特异能量比: 特异占比随深度衰减 = 真压缩
            hmu = out.mean(dim=1)                        # 逐样本时间均值 (n_in, D)
            h_pub = round(float(hmu.mean(0).norm().pow(2)
                                / hmu.norm(dim=-1).pow(2).mean().clamp_min(1e-12)), 4)
            # 绝对量: 公共分量是"稀释"还是"擦除"特异性? common/specific 的逐维 RMS
            nrm = out.norm(dim=1).pow(2).mean().sqrt()   # ||h|| RMS
            common = hmu.mean(0)                          # (D,) 跨输入公共向量
            spec = hmu - common                           # (n_in, D) 输入特异部分
            common_rms = round(float(common.norm() / nrm.clamp_min(1e-9)), 4)
            spec_rms = round(float(spec.norm(dim=-1).pow(2).mean().sqrt()
                                   / nrm.clamp_min(1e-9)), 4)
            rows.append({"layer": li, **br,
                         "out_temporal_energy_ratio": out_t_ratio,
                         "out_temporal_mean_cross_cos": out_t_cos,
                         "out_highpass_public_ratio": hp_pub,
                         "h_public_ratio": h_pub,
                         "h_input_specific_ratio": round(1.0 - h_pub, 4),
                         "h_common_rms": common_rms,
                         "h_specific_rms": spec_rms})

    print(json.dumps({"ckpt": args.ckpt, "domain": args.domain, "T": args.T,
                      "n_in": args.n_in, "layers": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()