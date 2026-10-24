"""层数扩展的代价律: 为什么"同参数下加层"会变慢, 以及慢在哪.

两个口径:
  affine  固定 dim, 扫 L -> 拟合 step_ms = c0*L + c1. c0 是每层固定开销
          (算子派发 + autograd 节点), c1 是与层数无关的部分. c0 >> 0 即
          "层数代价线性且截距大" = 扩展断崖的机制.
  budget  固定栈参数量, 扫 L -> 反解 dim, 测真实 step_ms. 展示同参数下
          层数增加如何在窄 dim 区把代价推上去.

每格一个独立子进程, 多轮交错, 报中位与配对.

用法:
  python -X utf8 experiments/_layer_scaling.py --mode affine --rounds 3
  python -X utf8 experiments/_layer_scaling.py --mode budget --budget 365844 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB = 8192

WORKER = r"""
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
torch.set_num_threads({threads})
from sunnynet.model import SunnyConfig, SunnyNet
cfg = json.loads(r'''{cfg}''')
c = SunnyConfig(vocab_size={vocab}, dim=cfg["dim"], inner=cfg["inner"],
                rank=cfg["rank"], layers=cfg["layers"], h_max=cfg["h_max"])
c.scan_chunk = 64
if cfg.get("ablation"):
    c.ablation = cfg["ablation"]
torch.manual_seed(1234)
m = SunnyNet(c)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
B, T = {B}, {T}
x = torch.randint(0, {vocab}, (B, T))
y = torch.randint(0, {vocab}, (B, T))
def step():
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, {vocab}), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
for _ in range({warm}):
    step()
ts = []
for _ in range({reps}):
    t0 = time.perf_counter()
    step()
    ts.append((time.perf_counter() - t0) * 1000.0)
tok = B * T
print(json.dumps({{"dim": cfg["dim"], "inner": cfg["inner"], "rank": cfg["rank"],
                  "layers": cfg["layers"],
                  "stack_params": m.num_params(non_embedding=True),
                  "total_params": m.num_params(),
                  "ms_step_median": round(statistics.median(ts), 3),
                  "ms_token_median": round(statistics.median(ts) / tok, 5)}}))
"""


def run_cell(cfg, threads, B, T, warm, reps):
    import statistics as _st  # noqa
    src = ("import statistics\n" + WORKER)
    code = src.format(root=str(ROOT).replace("\\", "/"), threads=threads,
                      cfg=json.dumps(cfg), vocab=VOCAB, B=B, T=T,
                      warm=warm, reps=reps)
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                         capture_output=True, text=True, cwd=str(ROOT))
    if out.returncode != 0:
        raise SystemExit(f"cell failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def solve_dim_for_budget(target, layers, ri, rr, h_max, threads):
    """反解 dim 使栈参数量最接近 target (inner=ri*dim, rank=rr*dim)."""
    lo, hi = 16, 4096
    best = None
    for _ in range(40):
        mid = (lo + hi) // 2
        cfg = {"dim": mid, "inner": max(8, round(ri * mid)),
               "rank": max(4, round(rr * mid)), "layers": layers, "h_max": h_max}
        n = run_cell(cfg, threads, 2, 64, 0, 1)["stack_params"]
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fit_affine(rows):
    """最小二乘 step_ms = c0*L + c1."""
    n = len(rows)
    sx = sum(r["layers"] for r in rows)
    sy = sum(r["ms_step_median"] for r in rows)
    sxx = sum(r["layers"] ** 2 for r in rows)
    sxy = sum(r["layers"] * r["ms_step_median"] for r in rows)
    den = n * sxx - sx * sx
    c0 = (n * sxy - sx * sy) / den
    c1 = (sy - c0 * sx) / n
    return round(c0, 3), round(c1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["affine", "budget"], default="affine")
    ap.add_argument("--L", default="1,2,3,4,6,8")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--budget", type=int, default=365844)
    ap.add_argument("--ratio-inner", type=float, default=1.5)
    ap.add_argument("--ratio-rank", type=float, default=0.5)
    ap.add_argument("--h-max", type=float, default=2048.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    Ls = [int(v) for v in args.L.split(",")]
    cells = {}
    for L in Ls:
        if args.mode == "affine":
            d = args.dim
            cfg = {"dim": d, "inner": max(8, round(args.ratio_inner * d)),
                   "rank": max(4, round(args.ratio_rank * d)), "layers": L,
                   "h_max": args.h_max}
        else:
            d, n = solve_dim_for_budget(args.budget, L, args.ratio_inner,
                                        args.ratio_rank, args.h_max, args.threads)
            cfg = {"dim": d, "inner": max(8, round(args.ratio_inner * d)),
                   "rank": max(4, round(args.ratio_rank * d)), "layers": L,
                   "h_max": args.h_max}
            print(f"[solve] L={L} -> dim={d} stack={n:,} (target {args.budget:,})")
        cells.setdefault(L, []).append(cfg)

    # 交错多轮: 每轮把全部 L 各跑一次, 取逐 L 中位
    acc = {L: [] for L in Ls}
    for r in range(args.rounds):
        for L in Ls:
            cfg = cells[L][0]
            rec = run_cell(cfg, args.threads, args.B, args.T, args.warm, args.reps)
            rec["round"] = r
            acc[L].append(rec)
            print(f"  r{r} L={L:2d} dim={rec['dim']:4d} "
                  f"stack={rec['stack_params']:>9,} "
                  f"{rec['ms_step_median']:.2f} ms/step "
                  f"{rec['ms_token_median']:.5f} ms/tok")

    rows = []
    for L in Ls:
        recs = acc[L]
        med = statistics.median(r["ms_step_median"] for r in recs)
        rows.append({"layers": L, "dim": recs[0]["dim"],
                     "inner": recs[0]["inner"], "rank": recs[0]["rank"],
                     "stack_params": recs[0]["stack_params"],
                     "ms_step_median": round(med, 3),
                     "ms_token_median": round(med / (args.B * args.T), 5),
                     "ms_step_rounds": [r["ms_step_median"] for r in recs]})

    res = {"mode": args.mode, "threads": args.threads, "B": args.B, "T": args.T,
           "ratio_inner": args.ratio_inner, "ratio_rank": args.ratio_rank,
           "rows": rows}
    if args.mode == "affine":
        c0, c1 = fit_affine(rows)
        res["fit_ms_step_per_layer_c0"] = c0
        res["fit_intercept_c1"] = c1
        base = next(r for r in rows if r["layers"] == 4)
        res["per_layer_cost_ms"] = round(base["ms_step_median"] / 4, 3)
        res["c0_over_per_layer_mean"] = round(c0 / (base["ms_step_median"] / 4), 3)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()