"""每层算子税的分段归因: 若要靠融合抹平"扩展断崖", 钱花在哪一段.

方法同 §14.7 的替换实验, 但覆盖块内全部阶段, 并把口径放在**每层边际成本**上:
把每个阶段的耗时替换为极简实现 (数值不等价, 只测各项耗时归零的收益上限),
测端到端训练步, 换算成"每层省多少 ms"。

阶段:
  local    深度可分离卷积历史 (F.pad + conv1d + transpose)
  compete  竞争门控 + 门控概率 (roll 家族)
  scan     logspace_scan
  readout  W_z + softwedge + norm_mix
  emit     W_c/W_s/b_gain/sn 组装
  norm     norm_in
判据: 某段归零后每层边际成本 (c0) 明显下降 => 该段是深度税主体, 融合它可削悬崖。

每格一个独立子进程。

用法: python -X utf8 experiments/_layer_tax.py [T] [rounds]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORKER = r"""
import json, sys, time, statistics
sys.path.insert(0, r"{root}")
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.set_num_threads({threads})
from sunnynet.model import SunnyConfig, SunnyNet

variant = "{variant}"
V, B, T, L = {vocab}, {B}, {T}, {L}
torch.manual_seed(1234)
c = SunnyConfig(vocab_size=V, dim=128, inner=192, rank=64, layers=L, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c)
m.train()

if variant != "base":
    if variant == "scan":
        import sunnynet.layers as SL
        SL.logspace_scan = (lambda u, p, s0, chunk=64, p_min_hint=None:
                            (u, u[:, -1]))
    for blk in m.blocks:
        if variant == "local":
            blk.dw = None
            blk.use_local_taps = False
        elif variant == "compete":
            blk._compete = lambda g: torch.ones_like(g)
        elif variant == "readout":
            blk._readout = lambda u, s: blk.W_z(u)
        elif variant == "emit":
            blk._emit = lambda u, states, s_in: blk.W_c(blk._readout(u, None))
        elif variant == "norm":
            blk.norm_in = nn.Identity()

opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
x = torch.randint(0, V, (B, T))
y = torch.randint(0, V, (B, T))
def step():
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
for _ in range(5):
    step()
ts = []
for _ in range(12):
    t0 = time.perf_counter(); step(); ts.append((time.perf_counter() - t0) * 1000.0)
print(json.dumps({{"variant": variant, "layers": L,
                  "ms_step_median": round(statistics.median(ts), 3)}}))
"""


def run_cell(variant, L, threads, B, T):
    code = WORKER.format(root=str(ROOT).replace("\\", "/"), threads=threads,
                         variant=variant, vocab=8192, B=B, T=T, L=L)
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                         capture_output=True, text=True, cwd=str(ROOT))
    if out.returncode != 0:
        raise SystemExit(f"cell {variant} L={L} failed:\n{out.stderr[-2500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    threads, B = 6, 8
    variants = ["base", "local", "compete", "scan", "readout", "emit", "norm"]
    Ls = [2, 6]

    acc = {(v, L): [] for v in variants for L in Ls}
    for _ in range(rounds):
        for L in Ls:
            for v in variants:
                acc[(v, L)].append(run_cell(v, L, threads, B, T)["ms_step_median"])

    res = {"T": T, "threads": threads, "B": B, "Ls": Ls, "variants": {}}
    base6 = statistics.median(acc[("base", 6)])
    base2 = statistics.median(acc[("base", 2)])
    for v in variants:
        m2 = statistics.median(acc[(v, 2)])
        m6 = statistics.median(acc[(v, 6)])
        c0 = (m6 - m2) / 4.0                          # 每层边际成本 (ms)
        res["variants"][v] = {
            "ms_L2": round(m2, 2), "ms_L6": round(m6, 2),
            "per_layer_ms": round(c0, 2),
            "step6_vs_base": round(m6 / base6, 4),
            "per_layer_vs_base": round(c0 / ((base6 - base2) / 4.0), 4),
        }
        print(f"{v:9s} L2={m2:7.2f} L6={m6:7.2f} per_layer={c0:6.2f}ms "
              f"rel={res['variants'][v]['per_layer_vs_base']:.3f}")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()