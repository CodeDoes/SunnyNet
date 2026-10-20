"""训练步速度规范对照: 多轮交错 + 每格独立子进程 + 取中位.

对照两个协议, 都报出来, 不做单一结论:
  A) 栈参数对齐 (项目既有协议): 各架构栈参数 ≈365,844
  B) 激活宽度对齐 (补充): sunnynet dim 压到同行 d=80

用法: python -X utf8 experiments/_train_bench.py [T] [rounds] [reps]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
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
CFG = {cfg}
if name.startswith("sunny"):
    d = CFG[name]
    c = SunnyConfig(vocab_size=V, dim=d["dim"], inner=d["inner"], rank=d["rank"], layers=d["layers"], h_max=2048.0)
    c.scan_chunk = 64
    m = SunnyNet(c)
else:
    ref = SunnyNet(SunnyConfig.small(V)); tgt = ref.num_params(non_embedding=True); del ref
    c, _, _ = build_linear_baseline(name, tgt, V, max_len=max(T, 2048), n_layers=6)
    m = _make(name, c)
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
print("CELL " + json.dumps({{"name": name, "T": T, "stack": stack, "train_ms": round(dt,3), "loss": round(lv,4)}}))
"""

CFG = {
    "sunny128":  {"dim": 128, "inner": 192, "rank": 64,  "layers": 4},
    "sunny112":  {"dim": 112, "inner": 192, "rank": 104, "layers": 4},
    "sunny104":  {"dim": 104, "inner": 192, "rank": 128, "layers": 4},
    "sunny96":   {"dim": 96,  "inner": 192, "rank": 152, "layers": 4},
    "sunny80":   {"dim": 80,  "inner": 160, "rank": 244, "layers": 4},
    "rwkv": None, "mamba": None, "h3": None,
}

cells = list(CFG)
acc = {c: [] for c in cells}
for rd in range(ROUNDS):
    for c in cells:
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), name=c, T=T, B=8,
                           reps=REPS, cfg=repr(CFG))
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"  {c} FAILED: {r.stderr[-300:]}")
            continue
        d = json.loads(line[0][5:])
        acc[c].append(d)
        print(f"  r{rd+1} {c:10s} T={T} stack={d['stack']:,} "
              f"{d['train_ms']}ms loss={d['loss']}")

print(f"\n=== T={T} 中位 ({ROUNDS} 轮) ===")
rows = []
for c in cells:
    if not acc[c]:
        continue
    ms = statistics.median([d["train_ms"] for d in acc[c]])
    rows.append({"name": c, "stack": acc[c][0]["stack"], "median_ms": round(ms, 1),
                 "rounds": [round(d["train_ms"], 1) for d in acc[c]]})
rows.sort(key=lambda r: r["median_ms"])
base = next((r["median_ms"] for r in rows if r["name"] == "sunny128"), None)
for r in rows:
    rel = f"{r['median_ms']/base:5.2f}x vs sunny128" if base else ""
    print(f"  {r['name']:10s} stack={r['stack']:>9,} median={r['median_ms']:8.1f}ms  {rel}")
Path("experiments/results/_train_bench.json").write_text(
    json.dumps({"T": T, "rounds": ROUNDS, "reps": REPS, "rows": rows},
               ensure_ascii=False, indent=2), encoding="utf-8")
print("RESULT " + json.dumps(rows, ensure_ascii=False))