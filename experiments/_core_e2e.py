"""融合 core 内核的训练步 A/B: 速度 + ppl.

每格独立子进程 (同进程构建多模型会污染耗时), 多轮交错 + 逐轮配对符号检验。

用法: python -X utf8 experiments/_core_e2e.py [T] [rounds] [threads]
      python -X utf8 experiments/_core_e2e.py --_cell {torch|kernel} T threads
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ["SUNNY_KERNEL"] = "{flag}"
import torch, torch.nn.functional as F
from sunnynet.model import SunnyConfig, SunnyNet
import sunnynet.kernels.core_auto as ca
ca.set_kernel_active({pyflag}, threads={threads})
torch.set_num_threads({threads})
torch.manual_seed(1234)
T, V, B = {T}, 8192, 8
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
def step():
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward(); opt.step()
    return float(loss.detach())
for _ in range(4): step()
reps = {reps}
t0 = time.perf_counter()
losses = [step() for _ in range(reps)]
dt = (time.perf_counter() - t0) / reps * 1000
print("CELL " + json.dumps({{"mode": "{mode}", "T": T, "threads": {threads},
      "step_ms": round(dt, 3), "loss_first": round(losses[0], 6),
      "loss_last": round(losses[-1], 6),
      "kernel": ca.kernel_active()}}))
'''


def run_cell(mode: str, T: int, threads: int, reps: int) -> dict:
    flag = "1" if mode == "kernel" else "0"
    code = CELL.format(root=str(ROOT), flag=flag, pyflag=str(mode == "kernel"),
                       threads=threads, T=T, reps=reps, mode=mode)
    env = dict(os.environ)
    env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:\devtools\w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT), env=env)
    for line in r.stdout.splitlines():
        if line.startswith("CELL "):
            return json.loads(line[5:])
    raise RuntimeError(f"cell {mode}/{T} 失败:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        print(run_cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), 20))
        return
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    reps = 20

    pair = {"torch": [], "kernel": []}
    for r in range(rounds):
        # 交错: 每轮先 torch 后 kernel, 下轮互换, 抵消热漂移
        order = ["torch", "kernel"] if r % 2 == 0 else ["kernel", "torch"]
        for mode in order:
            res = run_cell(mode, T, threads, reps)
            pair[mode].append(res["step_ms"])
            print(f"  r{r+1} {mode:7s} {res['step_ms']:8.3f} ms  "
                  f"loss {res['loss_first']:.4f}->{res['loss_last']:.4f}  "
                  f"kernel={res['kernel']}")

    import statistics as st
    t_med = st.median(pair["torch"])
    k_med = st.median(pair["kernel"])
    wins = sum(1 for a, b in zip(pair["kernel"], pair["torch"]) if a < b)
    ratios = [a / b for a, b in zip(pair["kernel"], pair["torch"])]
    # 配对比例中位数
    r_med = st.median(ratios)
    print(f"\nT={T} threads={threads} rounds={rounds}")
    print(f"torch  中位 {t_med:.2f} ms  {[round(v,2) for v in pair['torch']]}")
    print(f"kernel 中位 {k_med:.2f} ms  {[round(v,2) for v in pair['kernel']]}")
    print(f"中位比值 kernel/torch = {k_med/t_med:.4f} (加速 {t_med/k_med:.3f}x)")
    print(f"配对比值中位 = {r_med:.4f}   kernel 占优轮次 {wins}/{rounds}")
    out = ROOT / "experiments" / "results" / "_core_e2e.json"
    out.write_text(json.dumps({
        "T": T, "threads": threads, "rounds": rounds,
        "torch_ms": pair["torch"], "kernel_ms": pair["kernel"],
        "median_torch_ms": round(t_med, 3), "median_kernel_ms": round(k_med, 3),
        "median_ratio": round(k_med / t_med, 4),
        "paired_ratio_median": round(r_med, 4),
        "kernel_wins": wins,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()