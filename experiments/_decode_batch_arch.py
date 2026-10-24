"""批处理解码: B 增大后 O(1) 状态架构 vs Transformer 的单步成本.

§23.5 将"批处理解码"列为必测但未做。Transformer 单步注意力 B 增大后每行独立,
可被 BLAS 摊薄; SunnyNet 状态递推同样逐行独立, 但每行算子更多。
本脚本给出两条曲线随 B 的变化, 定位单步解码优势在批量下是否收窄。

口径: 每格独立子进程, 单线程, 先走满 prefill 建状态再测单步, 2 轮中位。
C 整模型后端只支持 B=1, 故批处理口径下 SunnyNet 走 torch.
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
V = 8192
TARGET = 365844
T = 256
BS = [1, 4, 8, 16, 32]

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
B, name, reps = {B}, "{name}", {reps}
torch.set_num_threads(1); torch.manual_seed(1234)
V = {V}; T = {T}
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params
from sunnynet.linear_baselines import build_linear_baseline, _make
ref = SunnyNet(SunnyConfig.small(V)); target = ref.num_params(non_embedding=True); del ref
if name == "sunny":
    c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
    c.scan_chunk = 64
    m = SunnyNet(c); stt = m.init_state(B)
elif name == "transformer":
    c, _, _ = solve_baseline_for_params(target, V, n_layers=6, n_heads=8, max_len=T + 8)
    m = TransformerLM(c); stt = None
else:
    c, _, _ = build_linear_baseline(name, target, V, max_len=T + 8, n_layers=6)
    m = _make(name, c); stt = m.init_state(B)
m.eval()
idx = torch.randint(0, V, (B, T))
with torch.no_grad():
    for t in range(T):
        _, stt = m.forward_step(idx[:, t], stt)
one = torch.randint(0, V, (B,))
with torch.no_grad():
    m.forward_step(one, stt)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); m.forward_step(one, stt); ts.append(time.perf_counter()-t0)
print("CELL " + json.dumps({{"ms": st.median(ts) * 1000}}))
'''


def cell(B, name):
    reps = 40 if B <= 8 else 20
    code = CELL.format(root=str(ROOT), B=B, name=name, reps=reps, V=V, T=T)
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=dict(os.environ))
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])["ms"]
    raise RuntimeError(f"{name}@B{B}: {(r.stderr or r.stdout or '')[-900:]}")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    names = ["sunny", "transformer", "rwkv"]
    acc = {(n, B): [] for n in names for B in BS}
    for rd in range(rounds):
        for B in BS:
            for n in names:
                v = cell(B, n)
                acc[(n, B)].append(v)
                print(f"  r{rd+1} {n:12s} B={B:<3d} {v:.4f} ms/step")
    tbl = {n: {B: st.median(acc[(n, B)]) for B in BS} for n in names}
    print(f"\n=== 单步解码 ms (prefill T={T}, 单线程) ===")
    print("arch".ljust(14) + "".join(f"B={B}".ljust(11) for B in BS))
    for n in names:
        print(f"  {n:12s}" + "".join(f"{tbl[n][B]:.4f}".ljust(11) for B in BS))
    print("\n=== 相对 Transformer 的比值 (>1 = Transformer 更慢) ===")
    for n in names:
        if n == "transformer":
            continue
        print(f"  {n:12s}" + "".join(f"{tbl['transformer'][B]/tbl[n][B]:.3f}".ljust(11) for B in BS))
    out = ROOT / "experiments/results/_decode_batch_arch.json"
    out.write_text(json.dumps({"T": T, "Bs": BS, "table": tbl, "rounds": rounds},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())