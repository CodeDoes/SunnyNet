"""优化 A/B 基准: 同机、同参数量、同序列长度下测量前向/反向耗时.

用于在改动算子后量化收益. 必须在独占环境运行; 并发负载会同时抬高两侧读数,
比值仍可比, 但绝对值应视为上界.
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
sys.path.insert(0, str(ROOT))

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def bench(model, B, T, n=4, warm=2, backward=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, model.cfg.vocab_size, (B, T))
    y = torch.randint(0, model.cfg.vocab_size, (B, T))

    def step():
        lg = model(x)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
        if backward:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return loss.item()

    for _ in range(warm):
        step()
    t0 = time.perf_counter()
    for _ in range(n):
        step()
    dt = (time.perf_counter() - t0) / n
    return dt, dt / (B * T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=["small", "proto20m"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lens", default="256,512,1024,2048")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from sunnynet.tokenizer import BPETokenizer
    tk = BPETokenizer.load(ROOT / "data" / "proc" / "tokenizer.json")
    vocab = tk.vocab_size
    mk = SunnyConfig.small if args.preset == "small" else SunnyConfig.proto20m

    rows = []
    for T in [int(v) for v in args.lens.split(",")]:
        m = SunnyNet(mk(vocab))
        B = args.batch if T <= 512 else max(1, args.batch * 512 // T)
        dt, mpt = bench(m, B, T, backward=not args.forward_only)
        rows.append({"seq_len": T, "batch": B, "seconds": round(dt, 4),
                     "ms_per_token": round(mpt * 1000, 5),
                     "params": m.num_params()})
        print(f"  T={T:5d} B={B}  {dt*1000:8.2f} ms/step  {mpt*1000:.4f} ms/tok")

    res = {"tag": args.tag, "preset": args.preset, "threads": args.threads,
           "backward": not args.forward_only, "rows": rows}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())