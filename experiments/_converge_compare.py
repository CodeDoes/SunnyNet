"""收敛态质量对照: 同 token 预算、同语料, baseline vs +关联记忆.

只看 loss 数字不足以判定"学到的能力"。本脚本在**同等训练预算**下分别训到收敛,
再统一用 train/evaluate.py 的三个探针测: 分域困惑度 / 长距离召回 / 生成样例.

用法:
  python -X utf8 experiments/_converge_compare.py --tag base
  python -X utf8 experiments/_converge_compare.py --tag assoc --slots 32 --m 16
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="sunnynet")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--mix", default="zh_ancient=1.0")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--assoc", action="store_true")
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--every", type=int, default=1,
                    help="assoc_every: 每 k 层开一个 (0=全开)")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from train.train import main as train_main

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    ckpt = runs / f"{args.tag}.pt"
    out = runs / f"{args.tag}.json"

    argv = [
        "--arch", args.arch, "--preset", args.preset, "--mix", args.mix,
        "--steps", str(args.steps), "--batch-size", str(args.B), "--seq-len", str(args.T),
        "--threads", str(args.threads), "--seed", str(args.seed),
        "--warmup", str(args.warmup), "--val-every", str(args.val_every),
        "--val-batches", "16", "--no-early-stop",
        "--save-ckpt", str(ckpt), "--out", str(out),
    ]
    if args.assoc:
        argv += ["--ablation",
                 f"assoc=1,assoc_slots={args.slots},assoc_m={args.m},assoc_every={args.every}"]
    sys.argv = ["train.py"] + argv
    print(f"=== 训练 tag={args.tag} arch={args.arch} steps={args.steps} "
          f"tokens={args.steps * args.B * args.T:,} assoc={args.assoc}")
    t0 = time.perf_counter()
    train_main()
    print(f"=== 训练完成 {args.tag} wall={time.perf_counter() - t0:.0f}s")
    print(json.dumps({"tag": args.tag, "ckpt": str(ckpt), "out": str(out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()