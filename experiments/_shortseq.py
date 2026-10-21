"""短序列倒挂: 训练态开销随 T 的实测, 对比三条配置.

回答: 质疑的 "T<384 慢于 Transformer" 在**训练态**(前向+反向+更新)是否仍存在,
以及 §16 的窄残差流优化对短序列的缓解幅度.

三条配置 (栈参数均 ≈365.8k):
  small     d128 i192 r64  L4  (旧基准)
  smalln80  d80  i160 r244 L4  (§16 窄配置)
  transformer (栈对齐, 6 层)

协议: batch=8, 每格独立子进程 (堆碎片污染), ms/token.
用法: python -X utf8 experiments/_shortseq.py [Ts] [rounds]
      python -X utf8 experiments/_shortseq.py --_cell <name> <T>
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
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params

name, T = "{name}", {T}
vocab = 8192
torch.set_num_threads(8); torch.manual_seed(1234)

ref = SunnyNet(SunnyConfig.small(vocab))
target = ref.num_params(non_embedding=True); del ref

CFG = {{
  "small":    lambda: SunnyConfig(vocab_size=vocab, dim=128, inner=192, rank=64,  layers=4, h_max=2048.0),
  "smalln80": lambda: SunnyConfig(vocab_size=vocab, dim=80,  inner=160, rank=244, layers=4, h_max=2048.0),
}}
if name in CFG:
    c = CFG[name](); c.scan_chunk = 64
    model = SunnyNet(c)
else:
    c, _, _ = solve_baseline_for_params(target, vocab, n_layers=6, n_heads=8,
                                        max_len=max(T, 4096))
    model = TransformerLM(c)

model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
B = 8
x = torch.randint(0, vocab, (B, T)); y = torch.randint(0, vocab, (B, T))

def step():
    lg = model(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); return float(loss.detach())

for _ in range(3):
    step()
reps = 12
t0 = time.perf_counter()
for _ in range(reps):
    step()
dt = (time.perf_counter() - t0) / reps
print("CELL " + json.dumps({{
    "name": name, "T": T, "stack": model.num_params(non_embedding=True),
    "ms_per_token": round(dt * 1000 / (B * T), 5), "step_ms": round(dt * 1000, 2)}}))
'''


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        name, T = sys.argv[2], int(sys.argv[3])
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), name=name, T=T)
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"FAIL {name}@{T}: {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["128", "256", "384", "512"])]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
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
                    print(f"  r{rd+1} {n}@{T} FAILED")
                    continue
                acc[(n, T)].append(json.loads(line[0]))
                print(f"  r{rd+1} {n:12s} T={T:<5d} "
                      f"{json.loads(line[0])['ms_per_token']:.5f} ms/tok")

    out = ROOT / "experiments" / "results" / "_shortseq.json"
    print(f"\n=== 训练态 ms/token 中位 ({rounds} 轮, batch=8) ===")
    print("config".ljust(13) + "".join(f"T={T}".ljust(11) for T in Ts))
    table = {}
    for n in names:
        row = []
        for T in Ts:
            v = statistics.median([d["ms_per_token"] for d in acc[(n, T)]]) \
                if acc[(n, T)] else None
            row.append(v)
        table[n] = row
        cells = "".join(f"{v:.5f}".ljust(11) if v else "n/a".ljust(11) for v in row)
        print(f"  {n:11s}" + cells)
    # 同比值
    print("\n=== 相对 Transformer 的比值 (>1 = 比 TF 慢) ===")
    print("config".ljust(13) + "".join(f"T={T}".ljust(11) for T in Ts))
    for n in names[:2]:
        row = []
        for i, T in enumerate(Ts):
            t = table["transformer"][i]
            row.append(table[n][i] / t if (table[n][i] and t) else None)
        cells = "".join(f"{v:.3f}".ljust(11) if v is not None else "n/a".ljust(11)
                        for v in row)
        print(f"  {n:11s}" + cells)

    out.write_text(json.dumps({"Ts": Ts, "rounds": rounds, "table": table},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()