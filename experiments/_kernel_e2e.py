"""端到端 A/B: SunnyBlock 用 AVX2 内核 vs torch scan 的训练步实测.

只在 scan 调用点切换, 其余完全一致 (等价性已由 tests 保证)。
每格独立子进程, 多轮交错取中位。

用法: python -X utf8 experiments/_kernel_e2e.py [T] [rounds] [threads]
      python -X utf8 experiments/_kernel_e2e.py --_cell <mode> <T> <threads>
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ.setdefault("SUNNY_W64", r"F:\devtools\w64devkit")
import torch

mode, T, th = "{mode}", {T}, {th}
torch.set_num_threads(th); torch.manual_seed(1234)

import sunnynet.layers as L
from sunnynet.model import SunnyConfig, SunnyNet

if mode == "kernel":
    from sunnynet.kernels.scan_auto import scan_auto
    L.logspace_scan = scan_auto        # 块内调用点
    L.scan_auto = scan_auto

cfgs = {{
  "small":   (128, 192,  64, 4),
  "smalln80": (80, 160, 244, 4),
}}
d, i, r, ln = cfgs["smalln80"]
c = SunnyConfig(vocab_size=8192, dim=d, inner=i, rank=r, layers=ln, h_max=2048.0)
c.scan_chunk = 64
model = SunnyNet(c).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
B = 8
x = torch.randint(0, 8192, (B, T)); y = torch.randint(0, 8192, (B, T))

def step():
    lg = model(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); return float(loss.detach())

for _ in range(3):
    step()
reps = 10
t0 = time.perf_counter()
last = 0.0
for _ in range(reps):
    last = step()
dt = (time.perf_counter() - t0) / reps
print("CELL " + json.dumps({{
    "mode": mode, "T": T, "threads": th,
    "step_ms": round(dt * 1000, 2),
    "ms_per_token": round(dt * 1000 / (B * T), 5),
    "loss": round(last, 6)}}))
'''


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        mode, T, th = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), mode=mode, T=T, th=th)
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"FAIL {mode}@{T}/t{th}: {r.stderr[-400:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    th = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    acc = {m: [] for m in ("torch", "kernel")}
    for rd in range(rounds):
        for m in ("torch", "kernel"):
            r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                "--_cell", m, str(T), str(th)],
                               capture_output=True, text=True, encoding="utf-8")
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not line:
                print(f"  r{rd+1} {m} FAILED"); continue
            d = json.loads(line[0]); acc[m].append(d)
            print(f"  r{rd+1} {m:7s} {d['step_ms']:8.2f} ms  "
                  f"{d['ms_per_token']:.5f} ms/tok loss={d['loss']}")

    out = ROOT / "experiments" / "results" / "_kernel_e2e.json"
    res = {}
    for m in acc:
        if acc[m]:
            res[m] = round(statistics.median([d["step_ms"] for d in acc[m]]), 2)
    print(f"\n=== 端到端训练步 (T={T}, threads={th}, smalln80, {rounds} 轮中位) ===")
    for m, v in res.items():
        print(f"  {m:8s} {v} ms")
    if len(res) == 2:
        print(f"  加速比 torch/kernel = {res['torch']/res['kernel']:.3f}x")
    out.write_text(json.dumps({"T": T, "threads": th, "rounds": rounds,
                               "median_step_ms": res}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()