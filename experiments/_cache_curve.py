"""缓存驻留对训练步的影响: 固定模型, 扫描 (B,T), 看 ms/token 是否偏离线性.

缓存论预测: 激活工作集超过 L3 (9 MB) 后, per-token 成本应上抬.
若 ms/token 在某个 (B,T) 出现极小值且该点工作集 < L3, 则"压工作集"是可兑现的杠杆.

用法: python -X utf8 experiments/_cache_curve.py --preset smalln80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from train.train import _preset_cfg  # noqa: E402

VOCAB = 8192
L3_KB = 9216


def act_bytes(cfg, B, T):
    """单层激活粗估: 残差流 + 扫描中间量 (若干倍 dim)."""
    per_tok = cfg.dim * 6 * 4          # 残差/候选/门/状态广播等
    return B * T * per_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smalln80")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--bt", default="8x64,8x128,8x256,8x512,4x512,16x256,4x256,2x1024")
    args = ap.parse_args()

    torch.set_num_threads(6)
    cfg = _preset_cfg(args.preset, VOCAB)
    cfg.scan_chunk = 64
    model = SunnyNet(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    rows = []
    for cell in args.bt.split(","):
        B, T = (int(v) for v in cell.split("x"))
        x = torch.randint(0, VOCAB, (B, T))
        y = torch.randint(0, VOCAB, (B, T))
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(
                model(x).reshape(-1, VOCAB), y.reshape(-1)).backward()
            opt.step()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(
                model(x).reshape(-1, VOCAB), y.reshape(-1)).backward()
            opt.step()
        dt = (time.perf_counter() - t0) / args.steps
        ms_tok = dt * 1000 / (B * T)
        rows.append({"B": B, "T": T, "tokens": B * T,
                     "ms_step": round(dt * 1000, 2),
                     "ms_token": round(ms_tok, 5),
                     "act_MB": round(act_bytes(cfg, B, T) / 1e6, 2),
                     "in_L3": act_bytes(cfg, B, T) / 1024 <= L3_KB})

    print(json.dumps({"preset": args.preset, "dim": cfg.dim, "layers": cfg.layers,
                      "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()