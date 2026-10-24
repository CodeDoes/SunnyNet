"""按层启用关联记忆的成本阶梯: 训练步与解码步各一格独立子进程.

关联记忆的成本 ∝ 开启层数. 本脚本扫 assoc_every ∈ {关, 4, 3, 2, 1}
(4=只在第 0 层开, 1=每层都开), 报训练 ms/token 与 B=1 解码 ms/token.

用法: python -X utf8 experiments/_assoc_layers.py [--slots 32] [--m 16]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r"""
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
torch.set_num_threads(6); torch.manual_seed(0)
V = 8192
ab = {ab}
cfg = SunnyConfig(vocab_size=V, dim=128, inner=192, rank=64, layers=4,
                  h_max=2048.0, ablation=ab)
cfg.scan_chunk = 64
m = SunnyNet(cfg)
n_assoc = sum(1 for b in m.blocks if b.assoc is not None)
params = m.num_params()

# 训练步: B8 x T256, 前向+反向
m = m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (8, 256)); y = torch.randint(0, V, (8, 256))
def tstep():
    l = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward(); opt.step()
for _ in range(4): tstep()
t0 = time.perf_counter()
for _ in range(12): tstep()
tr_ms = (time.perf_counter() - t0) / 12 * 1e3      # ms/step, step = 8*256 token
tr_per_tok = tr_ms / (8 * 256)

# 解码步: B=1 单步
m = m.eval()
st = m.init_state(1)
with torch.no_grad():
    for _ in range(4):
        _, st = m.forward_step(torch.randint(0, V, (1,)), st)
    t0 = time.perf_counter()
    for _ in range(60):
        _, st = m.forward_step(torch.randint(0, V, (1,)), st)
    dec_ms = (time.perf_counter() - t0) / 60 * 1e3

print("CELL " + json.dumps({{"tag": "{tag}", "n_assoc": n_assoc, "params": params,
    "train_ms": round(tr_ms,3), "train_per_tok": round(tr_per_tok,6),
    "decode_ms": round(dec_ms,4)}}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    # (tag, ablation dict)
    variants = [
        ("off", {}),
        ("every4", {"assoc": True, "assoc_every": 4, "assoc_slots": args.slots, "assoc_m": args.m}),
        ("every3", {"assoc": True, "assoc_every": 3, "assoc_slots": args.slots, "assoc_m": args.m}),
        ("every2", {"assoc": True, "assoc_every": 2, "assoc_slots": args.slots, "assoc_m": args.m}),
        ("every1", {"assoc": True, "assoc_every": 1, "assoc_slots": args.slots, "assoc_m": args.m}),
    ]
    acc = {t: [] for t, _ in variants}
    for rd in range(args.rounds):
        for tag, ab in variants:
            code = CELL.format(root=str(ROOT).replace("\\", "\\\\"),
                               ab=repr(ab), tag=tag)
            r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                               text=True, encoding="utf-8")
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
            if not line:
                print(f"  {tag} FAILED: {r.stderr[-400:]}")
                continue
            d = json.loads(line[0][5:])
            acc[tag].append(d)
            print(f"  r{rd+1} {tag:7s} assoc层={d['n_assoc']} "
                  f"训练={d['train_ms']:7.1f}ms({d['train_per_tok']*1e3:5.3f}µs/tok) "
                  f"解码={d['decode_ms']:7.4f}ms")

    rows = []
    for tag, _ in variants:
        if not acc[tag]:
            continue
        tr = statistics.median([d["train_ms"] for d in acc[tag]])
        tp = statistics.median([d["train_per_tok"] for d in acc[tag]])
        dc = statistics.median([d["decode_ms"] for d in acc[tag]])
        rows.append({"tag": tag, "n_assoc": acc[tag][0]["n_assoc"],
                     "params": acc[tag][0]["params"], "train_ms": round(tr, 2),
                     "train_us_per_tok": round(tp * 1e3, 4), "decode_ms": round(dc, 4)})
    base_tr = next((r["train_ms"] for r in rows if r["tag"] == "off"), None)
    base_dc = next((r["decode_ms"] for r in rows if r["tag"] == "off"), None)
    print(f"\n=== 中位 ({args.rounds} 轮, 每格独立子进程) ===")
    print(f"  {'tag':7s} {'层':>3s} {'参数':>9s} {'训练ms':>8s} {'倍率':>6s} "
          f"{'µs/tok':>7s} {'解码ms':>8s} {'倍率':>6s}")
    for r in rows:
        rr = f"{r['train_ms']/base_tr:.2f}x" if base_tr else ""
        dd = f"{r['decode_ms']/base_dc:.2f}x" if base_dc else ""
        print(f"  {r['tag']:7s} {r['n_assoc']:>3d} {r['params']:>9,} "
              f"{r['train_ms']:8.1f} {rr:>6s} {r['train_us_per_tok']:7.3f} "
              f"{r['decode_ms']:8.4f} {dd:>6s}")
    out = ROOT / "experiments" / "results" / "_assoc_layers.json"
    out.write_text(json.dumps({"slots": args.slots, "m": args.m, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()