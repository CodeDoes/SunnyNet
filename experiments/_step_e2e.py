"""推理态 (逐 token 解码) 内核 A/B: 单步融合内核对解码每步耗时的影响.

每格独立子进程, 多轮交错配对。对比对象是同一 SunnyNet 的内核路 vs torch 路,
不是与 Transformer 比 (那个见 _decode_bench.py)。

用法: python -X utf8 experiments/_step_e2e.py [Ts] [B] [rounds]
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ["SUNNY_KERNEL"] = "{flag}"
import torch
from sunnynet.model import SunnyConfig, SunnyNet
import sunnynet.kernels.core_auto as ca
ca.set_kernel_active({pyflag}, threads={threads})
torch.set_num_threads({threads}); torch.manual_seed(1234)
T, B, V = {T}, {B}, 8192
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).eval()
idx = torch.randint(0, V, (B, T))
def run():
    stt = m.init_state(B)
    with torch.no_grad():
        for t in range(T):
            lg, stt = m.forward_step(idx[:, t], stt)
    return lg
for _ in range(2): run()
reps = 3
t0 = time.perf_counter()
for _ in range(reps): run()
dt = (time.perf_counter() - t0) / reps
print("CELL " + json.dumps({{"mode": "{mode}", "T": T, "B": B, "threads": {threads},
      "ms_per_step": round(dt * 1000 / T, 5), "total_ms": round(dt * 1000, 3),
      "kernel": ca.kernel_active()}}))
'''


def run_cell(mode: str, T: int, B: int, threads: int) -> dict:
    code = CELL.format(root=str(ROOT), flag="1" if mode == "kernel" else "0",
                       pyflag=str(mode == "kernel"), threads=threads,
                       T=T, B=B, mode=mode)
    env = dict(os.environ)
    env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:\devtools\w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])
    raise RuntimeError(f"cell {mode} 失败:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")


def main():
    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["128", "256"])]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    threads = 8
    result = {}
    for T in Ts:
        pair = {"torch": [], "kernel": []}
        for r in range(rounds):
            for mode in (["torch", "kernel"] if r % 2 == 0 else ["kernel", "torch"]):
                res = run_cell(mode, T, B, threads)
                pair[mode].append(res["ms_per_step"])
                print(f"  T={T} r{r+1} {mode:7s} {res['ms_per_step']:.5f} ms/step "
                      f"(total {res['total_ms']:.1f} ms) kernel={res['kernel']}")
        t_med = st.median(pair["torch"])
        k_med = st.median(pair["kernel"])
        wins = sum(1 for a, b in zip(pair["kernel"], pair["torch"]) if a < b)
        ratios = [a / b for a, b in zip(pair["kernel"], pair["torch"])]
        result[T] = {"torch": t_med, "kernel": k_med, "ratio": k_med / t_med,
                     "speedup": t_med / k_med, "wins": wins, "rounds": rounds,
                     "paired_ratio_median": st.median(ratios)}
        print(f"  => T={T} B={B}: torch {t_med:.5f} | kernel {k_med:.5f} ms/step | "
              f"{t_med/k_med:.3f}x | 占优 {wins}/{rounds}\n")
    out = ROOT / "experiments" / "results" / "_step_e2e.json"
    out.write_text(json.dumps({"B": B, "threads": threads, "rounds": rounds,
                               "result": {str(k): v for k, v in result.items()}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()