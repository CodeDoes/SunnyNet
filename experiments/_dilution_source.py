"""层间稀释的来源归因: 公共分量从哪一支注入、是常量还是慢漂移.

残差流 out = W_c(y) + b_gain*sn + sfb + x. "跨输入公共分量" 是在输入维上取均值,
是线性算子, 故逐支的公共分量严格可加:
    common(out) = common(W_c(y)) + common(b_gain*sn) + common(sfb) + common(x)
据此把每层的公共分量**增量**归到具体支路。

同时区分两种公共分量:
  const   跨输入且跨 token 都不变的常量 (norm 的 center / 学习到的偏置);
  drift   跨输入共享但随 token 变化的慢分量 (状态积分器累积的 DC).
判据: 逐样本时间均值的跨样本余弦 (~1 = 纯常量, 明显 <1 = 慢漂移).

用法:
  python -X utf8 experiments/_dilution_source.py --ckpt experiments/ckpts_100m/sunnynet_p100mw.pt
  python -X utf8 experiments/_dilution_source.py --preset deepl8 --steps 2000 --tag dl8
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

VOCAB = 8192


def load(domain, split="train"):
    return np.memmap(ROOT / f"data/proc/tokens_{domain}.bin" if split == "train"
                     else ROOT / f"data/proc/val_{domain}.bin",
                     dtype=np.uint16, mode="r")


def pair_cos(mat):
    v = mat / mat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = v @ v.t()
    n = c.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return round(float(c[iu[0], iu[1]].mean()), 4)


def make_inputs(tr, T, n_in):
    starts = torch.randint(0, len(tr) - T - 1, (n_in,))
    return torch.stack([torch.from_numpy(np.asarray(tr[s:s + T], dtype=np.int64))
                        for s in starts])


def trace(model, xs, cfg):
    """逐层走真实块前向, 返回每层各支路的 (逐样本时间均值) 与公共分量."""
    L, D = cfg.layers, cfg.dim
    T = xs.shape[1]
    k = max(0, cfg.taps - 1)
    from sunnynet.ops import logspace_scan
    rows = []
    with torch.no_grad():
        h = model.embed(xs)
        s = torch.zeros(xs.shape[0], L, D)
        hist = torch.zeros(xs.shape[0], L, k, D)
        mem = None
        a0 = next((b.assoc for b in model.blocks if b.assoc is not None), None)
        if a0 is not None:
            mem = torch.zeros(xs.shape[0], L, a0.slots, a0.m)

        prev_common = h.mean(dim=1).mean(0)       # 嵌入层的公共分量
        rows.append({"layer": -1, "h_public_ratio": None,
                     "h_common_norm_rel": round(float(
                         prev_common.norm()
                         / h.mean(dim=1).norm(dim=-1).pow(2).mean().sqrt().clamp_min(1e-9)), 4),
                     "h_specific_norm_rel": None})
        for li, blk in enumerate(model.blocks):
            raw = blk.norm_in(h)
            a = blk._apply_local(raw, hist[:, li])
            u, gate, p = blk._core(a)
            states, _ = logspace_scan(u, p, s[:, li], chunk=blk.scan_chunk,
                                      p_min_hint=blk.p_min_bound)
            y = blk._readout(u, None)
            sn = blk.norm_s(states) if blk.norm_s is not None else states
            prev = blk._prev(states, s[:, li])
            if blk.norm_s is not None:
                prev = blk.norm_s(prev)
            sfb = (F.linear(prev, blk._Ws_eff()) if blk.circ is not None
                   else blk.W_s(prev))
            part_wc = blk.W_c(y)
            part_sn = blk.b_gain * sn
            part_sfb = sfb
            part_x = h
            core = part_wc + part_sn + part_sfb
            out = core + part_x

            if k:
                hist[:, li] = raw[:, T - k:T]
            s[:, li] = states[:, -1]
            h = out

            # 逐样本时间均值 (n_in,D); 其跨输入均值即公共分量
            def common(t):
                return t.mean(dim=1).mean(0)

            def tm(t):
                return t.mean(dim=1)

            hmu = tm(out)
            common_out = common(out)
            common_prev = prev_common
            inc = common_out - common_prev

            # 常量 vs 慢漂移: 逐样本时间均值的跨样本余弦
            drift_cos = pair_cos(hmu)
            # 各支路对公共分量增量的贡献占比 (D 维向量, 用与 inc 的内积归因)
            def share(t):
                c = common(t)
                denom = inc.norm().clamp_min(1e-12)
                return round(float((c @ inc / denom).item()), 4)

            # 状态 DC 强度
            s_dc = states.mean(dim=1)
            s_rms = states.norm(dim=1).pow(2).mean().sqrt()
            s_dc_rel = round(float(s_dc.norm(dim=-1).mean()
                                   / s_rms.clamp_min(1e-9)), 4)

            e_all = hmu.norm(dim=-1).pow(2).mean()
            pub_ratio = round(float(common_out.norm().pow(2)
                                    / e_all.clamp_min(1e-12)), 4)
            nrm = out.norm(dim=1).pow(2).mean().sqrt()
            spec = hmu - common_out
            rows.append({
                "layer": li,
                "h_public_ratio": pub_ratio,
                "h_input_specific_ratio": round(1.0 - pub_ratio, 4),
                "h_common_norm_rel": round(float(common_out.norm()
                                                 / nrm.clamp_min(1e-9)), 4),
                "h_specific_norm_rel": round(float(
                    spec.norm(dim=-1).pow(2).mean().sqrt()
                    / nrm.clamp_min(1e-9)), 4),
                "common_increment_norm": round(float(inc.norm()), 4),
                "common_growth_ratio": round(float(
                    common_out.norm() / common_prev.norm().clamp_min(1e-9)), 3),
                "common_direction_cos": round(float(
                    (common_out @ common_prev
                     / (common_out.norm() * common_prev.norm()).clamp_min(1e-12)).item()), 4),
                "common_is_const_vs_drift": drift_cos,
                "inc_share_wc": share(part_wc),
                "inc_share_sn": share(part_sn),
                "inc_share_sfb": share(part_sfb),
                "inc_share_x": share(part_x),
                "state_dc_rel": s_dc_rel,
                "state_rms": round(float(s_rms), 3),
            })
            prev_common = common_out
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--preset", default="")
    ap.add_argument("--ablation", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--domain", default="zh_ancient")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n-in", type=int, default=24)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    if args.ckpt:
        from train.evaluate import load_model
        model, cfg = load_model(args.ckpt)
    else:
        from train.train import _preset_cfg, lr_at
        from sunnynet.model import SunnyNet
        ablation = {}
        if args.ablation:
            for kv in args.ablation.split(","):
                kk, vv = kv.split("=")
                try:
                    iv = int(vv)
                except ValueError:
                    ablation[kk] = vv
                else:
                    ablation[kk] = bool(iv) if iv in (0, 1) else iv
        tr = load(args.domain)
        cfg = _preset_cfg(args.preset or "deepl8", VOCAB)
        cfg.scan_chunk = 64
        if ablation:
            cfg.ablation = dict(ablation)
        torch.manual_seed(1234)
        model = SunnyNet(cfg)
        if args.steps > 0:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            warm = max(1, args.steps // 20)
            for step in range(args.steps):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, args.steps, args.lr, warm)
                i = int(torch.randint(0, len(tr) - args.B * args.T - 2, (1,)))
                buf = np.asarray(tr[i:i + args.B * args.T + 1], dtype=np.int64)
                x = torch.from_numpy(buf[:-1]).reshape(args.B, args.T)
                yv = torch.from_numpy(buf[1:]).reshape(args.B, args.T)
                loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB),
                                                         yv.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()

    tr = load(args.domain)
    xs = make_inputs(tr, args.T, args.n_in)
    rows = trace(model, xs, cfg)

    tag = args.tag or (Path(args.ckpt).stem if args.ckpt else args.preset)
    rec = {"tag": tag, "preset": args.preset or "(ckpt)", "ablation": args.ablation,
           "layers": cfg.layers, "dim": cfg.dim, "T": args.T, "n_in": args.n_in,
           "stack_params": model.num_params(non_embedding=True), "rows": rows}
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()