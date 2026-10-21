"""真实训练步的线程数扫描: 每格独立子进程, 多轮交错取中位.

_train_bench.py 固定 8 线程, 本脚本回答"8 是否最优".
用法: python -X utf8 experiments/_thread_scan.py [T] [rounds] [reps]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
THREADS = [4, 6, 8, 12]

CELL = """
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
torch.set_num_threads({th})
torch.manual_seed(1234)
T, B, reps = {T}, 8, {reps}
V = 8192
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
def step():
    l = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True); l.backward(); opt.step(); return float(l.detach())
for _ in range(5): lv = step()
t0 = time.perf_counter()
for _ in range(reps): lv = step()
dt = (time.perf_counter() - t0) / reps * 1e3
print("CELL " + json.dumps({{"th": {th}, "T": T, "train_ms": round(dt, 3), "loss": round(lv, 4)}}))
"""


def run_cell(th: int) -> dict | None:
    code = CELL.format(root=ROOT, th=th, T=T, reps=REPS)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"  th={th} 失败: {r.stderr.strip()[-200:]}", file=sys.stderr)
        return None
    for line in r.stdout.splitlines():
        if line.startswith("CELL "):
            return json.loads(line[5:])
    return None


def main():
    print(f"T={T} rounds={ROUNDS} reps={REPS} threads={THREADS}")
    samples: dict[int, list[float]] = {t: [] for t in THREADS}
    for rd in range(ROUNDS):
        for th in THREADS:
            r = run_cell(th)
            if r:
                samples[th].append(r["train_ms"])
                print(f"  轮{rd + 1} th={th:2d}  {r['train_ms']:.3f} ms/step")
    print("\n=== 中位 ===")
    med = {}
    for th in THREADS:
        v = samples[th]
        if not v:
            continue
        med[th] = statistics.median(v)
        spread = (max(v) - min(v)) / statistics.median(v) * 100
        print(f"  th={th:2d}  中位 {med[th]:.3f} ms  轮间极差 {spread:.1f}%")
    if med:
        best = min(med, key=med.get)
        base = med.get(8)
        print(f"\n最快: {best} 线程 ({med[best]:.3f} ms)")
        if base:
            print(f"相对 8 线程: {med[best] / base:.3f}x "
                  f"({(base / med[best] - 1) * 100:+.1f}% 提速)")
    print("RAW " + json.dumps({str(k): v for k, v in samples.items()}))


if __name__ == "__main__":
    main()