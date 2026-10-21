"""批次扫描: ms/token 随 batch 的变化, 检验矩阵效率是否被小形状拖累.

每格独立子进程. 用法: python -X utf8 experiments/_batch_scan.py [T] [rounds] [reps]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
BATCHES = [4, 8, 16, 32]

CELL = """
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
torch.set_num_threads(8); torch.manual_seed(1234)
T, B, reps = {T}, {B}, {reps}
V = 8192
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
def step():
    l = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward(); opt.step(); return float(l.detach())
for _ in range(3): lv = step()
t0 = time.perf_counter()
for _ in range(reps): lv = step()
dt = (time.perf_counter() - t0) / reps
print("CELL " + json.dumps({{"B": B, "T": T, "ms_step": round(dt * 1e3, 2),
    "ms_per_token": round(dt * 1e3 / (B * T), 5), "loss": round(lv, 3)}}))
"""


def run_cell(b: int) -> dict | None:
    code = CELL.format(root=ROOT, T=T, B=b, reps=REPS)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  B={b} 失败: {r.stderr.strip()[-200:]}", file=sys.stderr)
        return None
    for line in r.stdout.splitlines():
        if line.startswith("CELL "):
            return json.loads(line[5:])
    return None


def main():
    print(f"T={T} rounds={ROUNDS} reps={REPS} batches={BATCHES}")
    samples: dict[int, list[float]] = {b: [] for b in BATCHES}
    for rd in range(ROUNDS):
        for b in BATCHES:
            r = run_cell(b)
            if r:
                samples[b].append(r["ms_per_token"])
                print(f"  轮{rd + 1} B={b:2d}  {r['ms_step']:8.2f} ms/step  "
                      f"{r['ms_per_token']:.5f} ms/token")
    print("\n=== ms/token 中位 ===")
    med = {}
    for b in BATCHES:
        v = samples[b]
        if v:
            med[b] = statistics.median(v)
            print(f"  B={b:2d}  {med[b]:.5f} ms/token   "
                  f"吞吐 {1 / med[b] * 1000:7.0f} token/s")
    if med:
        b0 = min(med)
        best = max(med, key=lambda k: 1 / med[k])
        print(f"\n最省: B={best}  (相对 B={b0} 省 {(med[b0] / med[best] - 1) * 100:+.1f}%)")
    print("RAW " + json.dumps({str(k): v for k, v in samples.items()}))


if __name__ == "__main__":
    main()