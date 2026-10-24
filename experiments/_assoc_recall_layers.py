"""召回容量 vs 开启层数: 几层关联记忆才够.

`_assoc_layers.py` 已给出成本阶梯 (每层 +22% 训练耗时). 本脚本扫层数 x 键对数,
报 MQAR 末位召回准确率, 用于判定"少开几层保不保得住容量".

用法: python -X utf8 experiments/_assoc_recall_layers.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet

N_SYM = 64


def make_batch(B, n_pairs, dev):
    k = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    v = torch.randint(0, N_SYM, (B, n_pairs), device=dev)
    qi = torch.randint(0, n_pairs, (B,), device=dev)
    x = torch.stack([k, v], dim=2).reshape(B, 2 * n_pairs)
    q = k.gather(1, qi[:, None])
    x = torch.cat([x, q], dim=1)
    y = v.gather(1, qi[:, None]).squeeze(1)
    return x, y


def train_eval(model, n_pairs, steps, B, dev, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = make_batch(B, n_pairs, dev)
        loss = F.cross_entropy(model(x)[:, -1], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        x, y = make_batch(256, n_pairs, dev)
        acc = float((model(x)[:, -1].argmax(-1) == y).float().mean())
    model.train()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="2,4,8")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cpu")
    torch.set_num_threads(6)

    variants = [
        ("off", {}),
        ("every4", {"assoc": True, "assoc_every": 4, "assoc_slots": args.slots, "assoc_m": args.m}),
        ("every2", {"assoc": True, "assoc_every": 2, "assoc_slots": args.slots, "assoc_m": args.m}),
        ("every1", {"assoc": True, "assoc_every": 1, "assoc_slots": args.slots, "assoc_m": args.m}),
    ]
    rows = []
    for n in [int(v) for v in args.pairs.split(",")]:
        r = {"pairs": n, "seq_len": 2 * n + 1}
        for name, ab in variants:
            torch.manual_seed(0)
            cfg = SunnyConfig(vocab_size=N_SYM, dim=args.dim, inner=args.dim * 2,
                              rank=args.dim // 2, layers=args.layers, ablation=ab)
            m = SunnyNet(cfg)
            na = sum(1 for b in m.blocks if b.assoc is not None)
            t0 = time.perf_counter()
            acc = train_eval(m, n, args.steps, args.B, dev)
            r[f"{name}_acc"] = round(acc, 4)
            r[f"{name}_n"] = na
            r[f"{name}_params"] = m.num_params()
            r[f"{name}_s"] = round(time.perf_counter() - t0, 1)
            print(f"  pairs={n} {name:7s} 层={na} acc={acc:.3f} ({r[f'{name}_s']}s)")
        rows.append(r)
    print("\n" + json.dumps({"dim": args.dim, "layers": args.layers,
                             "steps": args.steps, "rows": rows}, ensure_ascii=False))
    out = ROOT / "experiments" / "results" / "_assoc_recall_layers.json"
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()