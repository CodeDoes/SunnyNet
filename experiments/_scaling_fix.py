"""同参数预算下的 (层数, 宽度) 族对照: 深度是否买到质量, 还是只买到延迟.

动机: 参数对齐时 SunnyNet 层数远超同行 (低秩 + 窄 dim -> 单层参数少 -> 被迫堆深),
而每层是一笔固定的算子派发税 -> 深 + 窄 = 同参数下更慢。本脚本在**固定栈参数量**
下枚举若干 (L, dim, inner, rank) 组合, 测训练步延迟, 可选跑短程质量 A/B。

判据:
  同预算下 "浅而宽" 若延迟更低且 ppl 不劣 => 深度不是质量来源, 是派发税来源。
  可解耦: 把预算放进 matmul 侧 (inner/rank), 而不是层数侧。

每格一个独立子进程, 多轮交错取中位。

用法:
  python -X utf8 experiments/_scaling_fix.py --budget 365844 --rounds 3
  python -X utf8 experiments/_scaling_fix.py --budget 16755230 --rounds 2 --steps 0
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
import json, math, sys, time, statistics
sys.path.insert(0, r"{root}")
import numpy as np
import torch
torch.set_num_threads({threads})
from sunnynet.model import SunnyConfig, SunnyNet
cfg = json.loads(r'''{cfg}''')
c = SunnyConfig(vocab_size={vocab}, dim=cfg["dim"], inner=cfg["inner"],
                rank=cfg["rank"], layers=cfg["layers"], h_max=cfg["h_max"])
c.scan_chunk = 64
torch.manual_seed({seed})
m = SunnyNet(c)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
B, T = {B}, {T}
tr = np.memmap(r"{tokens}", dtype=np.uint16, mode="r")

def get_batch():
    i = int(torch.randint(0, len(tr) - B * T - 2, (1,)))
    buf = np.asarray(tr[i:i + B * T + 1], dtype=np.int64)
    return (torch.from_numpy(buf[:-1]).reshape(B, T),
            torch.from_numpy(buf[1:]).reshape(B, T))

steps = {steps}
warm = max(1, min(steps // 20, 40)) if steps else 0
if steps > 0:
    for s in range(steps):
        x, y = get_batch()
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, {vocab}), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    # 验证 ppl
    va = np.memmap(r"{val_tokens}", dtype=np.uint16, mode="r")
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(16):
            i = int(torch.randint(0, len(va) - T - 2, (1,)))
            w = torch.from_numpy(np.asarray(va[i:i + T + 1], dtype=np.int64))
            lg = m(w[None, :-1])
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, {vocab}), w[1:], reduction="sum"))
            n += T
    ppl = round(math.exp(min(tot / n, 20)), 2)
    m.train()
else:
    ppl = None

x, y = get_batch()
def step():
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, {vocab}), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
for _ in range({warm_perf}):
    step()
ts = []
for _ in range({reps}):
    t0 = time.perf_counter(); step(); ts.append((time.perf_counter() - t0) * 1000.0)
tok = B * T
print(json.dumps({{"dim": cfg["dim"], "inner": cfg["inner"], "rank": cfg["rank"],
                  "layers": cfg["layers"], "step_scale": cfg.get("step_scale", 1),
                  "stack_params": m.num_params(non_embedding=True),
                  "total_params": m.num_params(),
                  "ms_step_median": round(statistics.median(ts), 3),
                  "ms_token_median": round(statistics.median(ts) / tok, 5),
                  "val_ppl_zh": ppl}}))
"""


def run_cell(cfg, threads, B, T, steps, reps, warm_perf, seed=1234):
    code = WORKER.format(root=str(ROOT).replace("\\", "/"), threads=threads,
                         cfg=json.dumps(cfg), vocab=VOCAB, B=B, T=T, steps=steps,
                         warm=int(warm_perf), reps=reps, warm_perf=warm_perf,
                         seed=seed,
                         tokens=str(ROOT / "data/proc/tokens_zh_ancient.bin"),
                         val_tokens=str(ROOT / "data/proc/val_zh_ancient.bin"))
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                         capture_output=True, text=True, cwd=str(ROOT))
    if out.returncode != 0:
        raise SystemExit(f"cell failed:\n{out.stderr[-2500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _stack_params_meta(dim, inner, rank, layers, h_max):
    """meta device 计数: 不分配内存, 实例化取名参数量."""
    import torch
    from sunnynet.model import SunnyConfig, SunnyNet
    with torch.device("meta"):
        c = SunnyConfig(vocab_size=VOCAB, dim=dim, inner=inner, rank=rank,
                        layers=layers, h_max=h_max)
        m = SunnyNet(c)
    return m.num_params(non_embedding=True)


def scale_to_budget(dim, inner, rank, layers, target, h_max=2048.0):
    """按比例缩放一组 (dim,inner,rank) 到目标栈参数量 (meta device 计数)."""
    n = _stack_params_meta(dim, inner, rank, layers, h_max)
    f = (target / n) ** 0.5
    out = {"dim": max(16, round(dim * f)), "inner": max(8, round(inner * f)),
           "rank": max(4, round(rank * f)), "layers": layers, "h_max": h_max}
    # 一轮修正: 按实际计数微调 dim
    for _ in range(8):
        got = _stack_params_meta(out["dim"], out["inner"], out["rank"],
                                 out["layers"], out["h_max"])
        if abs(got / target - 1) < 0.01:
            break
        f = (target / got) ** 0.5
        out["dim"] = max(16, round(out["dim"] * f))
        out["inner"] = max(8, round(inner * (out["dim"] / dim)))
        out["rank"] = max(4, round(rank * (out["dim"] / dim)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=365844)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600, help="0=只测速度")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warm-perf", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--h-max", type=float, default=2048.0)
    ap.add_argument("--seeds", default="1234", help="逗号分隔的种子列表")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # 族: 同参数预算下, 从"浅而宽"到"深而窄"
    families = {
        "L3 wide":  (3, 224, 384, 160),
        "L4 base":  (4, 128, 192, 64),
        "L4 wide":  (4, 176, 288, 112),
        "L6 narrow": (6, 96, 152, 96),
        "L6 wide":  (6, 144, 256, 96),
        "L8 narrow": (8, 80, 128, 160),
    }
    cfgs = {}
    for name, (L, d, i, r) in families.items():
        c = scale_to_budget(d, i, r, L, args.budget, args.h_max)
        n = run_cell(c, args.threads, args.B, args.T, 0, 1, 0)
        print(f"[cfg] {name:10s} L={c['layers']} d={c['dim']:4d} i={c['inner']:4d} "
              f"r={c['rank']:3d} stack={n['stack_params']:,} "
              f"(target {args.budget:,}, dev {(n['stack_params']/args.budget-1)*100:+.1f}%)")
        cfgs[name] = c

    acc = {k: [] for k in cfgs}
    for r in range(args.rounds):
        for name, c in cfgs.items():
            seed = seeds[r % len(seeds)]
            rec = run_cell(c, args.threads, args.B, args.T, args.steps,
                           args.reps, args.warm_perf, seed=seed)
            rec["round"] = r
            rec["seed"] = seed
            acc[name].append(rec)
            print(f"  r{r} seed{seed} {name:10s} L={rec['layers']} d={rec['dim']:4d} "
                  f"{rec['ms_step_median']:7.2f} ms/step "
                  f"{rec['ms_token_median']:.5f} ms/tok "
                  f"ppl={rec['val_ppl_zh']}")

    rows = []
    for name, recs in acc.items():
        med = statistics.median(x["ms_step_median"] for x in recs)
        ppls = [x["val_ppl_zh"] for x in recs if x["val_ppl_zh"] is not None]
        rows.append({"family": name, "layers": recs[0]["layers"],
                     "dim": recs[0]["dim"], "inner": recs[0]["inner"],
                     "rank": recs[0]["rank"],
                     "stack_params": recs[0]["stack_params"],
                     "ms_step_median": round(med, 3),
                     "ms_token_median": round(med / (args.B * args.T), 5),
                     "ms_step_rounds": [x["ms_step_median"] for x in recs],
                     "val_ppl_zh": (round(statistics.median(ppls), 2) if ppls else None)})
    rows.sort(key=lambda r: r["ms_step_median"])
    res = {"budget": args.budget, "threads": args.threads, "B": args.B, "T": args.T,
           "steps": args.steps, "rows": rows}

    if len(rows) >= 2:
        fast, slow = rows[0], rows[-1]
        res["fastest"] = fast["family"]
        res["speedup_fastest_vs_slowest"] = round(slow["ms_step_median"]
                                                  / fast["ms_step_median"], 3)
    # 浅宽 vs 深窄: 同预算下延迟与质量
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {p}")


if __name__ == "__main__":
    main()