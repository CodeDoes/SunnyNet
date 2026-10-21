"""推理态 (逐 token 自回归解码) 开销实测: SunnyNet vs Transformer.

背景: 批评称 "T<384 慢于 Transformer, 限制其在短句分类/对话回复中的应用".
但 §2/§19 全部数据都是**训练态或整段前向**; 而 "对话回复" 的真实部署形态是
**逐 token 解码** —— 此前双方都未在此口径下比过 (Transformer 的 forward_step
在本仓库原是 NotImplementedError, 本轮才实现 KV cache).

关键机制差异 (决定解码态结论可能与训练态相反):
  - Transformer 解码: 每步要对历史做 O(t) 注意力 (KV cache 省了重算 K/V, 但
    q·k 仍是 O(t)), 单步访存随 t 线性增长.
  - SunnyNet 解码: 状态 (B,L,D) 定长, 单步开销**与 t 无关**, 恒为 O(1).

协议: eval + no_grad, batch=1, 逐步解码 T 步, 报 ms/token (每步耗时).
每格独立子进程, 多轮交错取中位.

用法: python -X utf8 experiments/_decode_bench.py [Ts] [rounds]
      python -X utf8 experiments/_decode_bench.py --_cell <name> <T>
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SPEC = {"small": (128, 192, 64, 4), "smalln80": (80, 160, 244, 4)}
TARGET = 365844

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params

name, T, B = "{name}", {T}, {B}
vocab = 8192
torch.set_num_threads(8); torch.manual_seed(1234)
ref = SunnyNet(SunnyConfig.small(vocab))
target = ref.num_params(non_embedding=True); del ref

if name in {spec}:
    d, i, r, L = {spec}[name]
    c = SunnyConfig(vocab_size=vocab, dim=d, inner=i, rank=r, layers=L, h_max=2048.0)
    c.scan_chunk = 64
    model = SunnyNet(c)
else:
    c, _, _ = solve_baseline_for_params(target, vocab, n_layers=6, n_heads=8,
                                        max_len=max(T, 4096))
    model = TransformerLM(c)
model.eval()
idx = torch.randint(0, vocab, (B, T))

def run():
    st = None
    if name in {spec}:
        st = model.init_state(B)
    with torch.no_grad():
        for t in range(T):
            lg, st = model.forward_step(idx[:, t], st)
    return lg

for _ in range(2):
    run()
reps = 3
t0 = time.perf_counter()
for _ in range(reps):
    run()
dt = (time.perf_counter() - t0) / reps
print("CELL " + json.dumps({{
    "name": name, "T": T, "B": B, "steps": T,
    "ms_per_step": round(dt * 1000 / T, 5),
    "ms_per_token": round(dt * 1000 / (B * T), 5),
    "total_ms": round(dt * 1000, 2),
    "stack": model.num_params(non_embedding=True)}}))
'''


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        name, T = sys.argv[2], int(sys.argv[3])
        B = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), name=name, T=T, B=B,
                           spec=repr(SPEC))
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"FAIL {name}@{T}: {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["32", "64", "128", "256", "512", "1024"])]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    names = ["small", "smalln80", "transformer"]

    acc = {(n, T): [] for n in names for T in Ts}
    for rd in range(rounds):
        for T in Ts:
            for n in names:
                r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                    "--_cell", n, str(T)],
                                   capture_output=True, text=True, encoding="utf-8")
                line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                if not line:
                    print(f"  r{rd+1} {n}@{T} FAILED"); continue
                acc[(n, T)].append(json.loads(line[0]))
                print(f"  r{rd+1} {n:12s} T={T:<5d} "
                      f"{json.loads(line[0])['ms_per_step']:.5f} ms/step")

    out = ROOT / "experiments" / "results" / "_decode_bench.json"
    print(f"\n=== 解码态 ms/step 中位 ({rounds} 轮, batch=1) ===")
    print("config".ljust(13) + "".join(f"T={T}".ljust(11) for T in Ts))
    table = {}
    for n in names:
        row = []
        for T in Ts:
            v = statistics.median([d["ms_per_step"] for d in acc[(n, T)]]) \
                if acc[(n, T)] else None
            row.append(v)
        table[n] = row
        print(f"  {n:11s}" + "".join(f"{v:.5f}".ljust(11) if v else "n/a".ljust(11)
                                     for v in row))
    print("\n=== 相对 Transformer 的比值 (>1 = 比 TF 慢) ===")
    for n in names[:-1]:
        row = []
        for i, T in enumerate(Ts):
            t = table["transformer"][i]
            v = table[n][i]
            row.append(v / t if (v and t) else None)
        print(f"  {n:11s}" + "".join(f"{v:.3f}".ljust(11) if v is not None
                                     else "n/a".ljust(11) for v in row))

    out.write_text(json.dumps({"Ts": Ts, "rounds": rounds, "table": table},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()