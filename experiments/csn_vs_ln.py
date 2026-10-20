"""决定性头对头: 修正后的 CenteredScaleNorm vs LayerNorm, 均带残差旁路.

动机: 早期实测 ln_resid(LN+残差) ppl≈55.3, 而 residual(CSN+残差) ppl≈69,
差距 ~12 ppl 全归咎于归一化层。但该 LN 数据在旧 CSN(未真正归一化)下测得,
CSN 修正后结论需重测。本脚本在同一协议、多子种下直接对撞两者。

用法: python experiments/csn_vs_ln.py --steps 600
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = ("smin50", "smin100", "ln_resid")


def _run_cell(variant: str, steps: int, seed: int, threads: int) -> dict | None:
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "experiments" / "_cell_ab.py"),
         variant, str(steps), str(seed), str(threads)],
        capture_output=True, text=True, cwd=str(ROOT))
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print(f"{variant}/{seed}: 失败\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
        return None
    return json.loads(line[-1][7:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", default="1234,2024,3407")
    ap.add_argument("--variants", default=",".join(CANDIDATES))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "csn_vs_ln.json"))
    a = ap.parse_args()

    rows = []
    for v in a.variants.split(","):
        ppls = []
        for s in a.seeds.split(","):
            d = _run_cell(v, a.steps, int(s), a.threads)
            if d:
                rows.append(d)
                ppls.append(d["ppl"])
                print(f"{v:10s} seed={s}  ppl={d['ppl']:8.3f}", flush=True)
        if ppls:
            print(f"  --> {v:10s} 均值 ppl={sum(ppls) / len(ppls):.3f}\n", flush=True)

    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"-> {a.out}")