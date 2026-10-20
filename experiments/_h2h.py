"""head-to-head: sunny80 vs rwkv, 逐轮配对, 每格独立子进程.

给出配对比值分布与符号检验, 比单一中位数更能支撑"持平/超越"的判断.
用法: python -X utf8 experiments/_h2h.py [T] [rounds] [reps]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 25

CELL = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.linear_baselines import _make, build_linear_baseline
torch.set_num_threads(8); torch.manual_seed(1234)
name, T, B, reps = "{name}", {T}, 8, {reps}
V = 8192
if name == "sunny80":
    c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
    c.scan_chunk = 64
    m = SunnyNet(c)
else:
    ref = SunnyNet(SunnyConfig.small(V)); tgt = ref.num_params(non_embedding=True); del ref
    cc, _, _ = build_linear_baseline("rwkv", tgt, V, max_len=max(T, 2048), n_layers=6)
    m = _make("rwkv", cc)
m = m.train(); stack = m.num_params(non_embedding=True)
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
def step():
    l = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward(); opt.step(); return float(l.detach())
for _ in range(5): lv = step()
t0 = time.perf_counter()
for _ in range(reps): lv = step()
dt = (time.perf_counter() - t0) / reps * 1e3
print("CELL " + json.dumps({{"name": name, "ms": round(dt,3), "loss": round(lv,4)}}))
"""


def run(name):
    code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), name=name, T=T, B=8, reps=REPS)
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
    if not line:
        raise RuntimeError(f"{name} failed: {r.stderr[-300:]}")
    return json.loads(line[0][5:])


pairs = []
for rd in range(ROUNDS):
    a = run("sunny80")
    b = run("rwkv")
    pairs.append((a["ms"], b["ms"]))
    print(f"  r{rd+1} sunny80={a['ms']:7.1f}ms  rwkv={b['ms']:7.1f}ms  "
          f"ratio(s/r)={a['ms']/b['ms']:.3f}")

ratios = [s / r for s, r in pairs]
wins = sum(1 for s, r in pairs if s < r)
s_ms = [s for s, _ in pairs]
r_ms = [r for _, r in pairs]
to = T * 8
print(f"\n=== T={T} head-to-head ({ROUNDS} 轮) ===")
print(f"  sunny80 中位 {statistics.median(s_ms):7.1f}ms ({statistics.median(s_ms)/to:.4f} ms/tok)")
print(f"  rwkv    中位 {statistics.median(r_ms):7.1f}ms ({statistics.median(r_ms)/to:.4f} ms/tok)")
print(f"  配对比值 s/r 中位 {statistics.median(ratios):.3f}  均值 {statistics.mean(ratios):.3f}")
print(f"  sunny80 更快轮数 {wins}/{ROUNDS}")
out = {"T": T, "rounds": ROUNDS,
       "sunny80_ms": s_ms, "rwkv_ms": r_ms, "ratios": [round(x, 4) for x in ratios],
       "median_ratio": round(statistics.median(ratios), 4),
       "wins_sunny80": wins}
Path(f"experiments/results/_h2h_T{T}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("RESULT " + json.dumps(out, ensure_ascii=False))