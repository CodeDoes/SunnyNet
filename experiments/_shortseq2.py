"""短序列倒挂探底: 极短 T / 线程数 / 更窄残差流 三个旋钮的训练态实测.

§19 只覆盖 T>=128、batch=8、8 线程. 本节补三块此前未测的量:
  1. 极短序列 T=32/64/96 (常规短句分类/意图识别的真实长度区间)
  2. 线程数 (短序列上小张量多线程同步开销可能超过并行收益, 未测过)
  3. 更窄残差流 (d64/d72, 逐元素算子按激活宽度付费)

所有配置栈参数量对齐 small (~365.8k), 对齐靠 meta device 实测计数.

协议: batch=8 训练步 (前向+反向+AdamW), 每格独立子进程, 多轮交错取中位.
用法: python -X utf8 experiments/_shortseq2.py [Ts] [threads] [rounds]
      python -X utf8 experiments/_shortseq2.py --_cell <name> <T> <threads>
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 栈参数量对齐 small: dim128 inner192 rank64 L4 = 365844 (meta device 实测)
TARGET = 365844

CFG_SPEC = {
    # name: (dim, inner, rank, layers)
    "small":    (128, 192, 64, 4),
    "smalln96": (96, 192, 152, 4),
    "smalln80": (80, 160, 244, 4),
    "smalln72": (72, 144, 300, 4),
    "smalln64": (64, 128, 368, 4),
}

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params

name, T, th = "{name}", {T}, {th}
vocab = 8192
torch.set_num_threads(th); torch.manual_seed(1234)

ref = SunnyNet(SunnyConfig.small(vocab))
target = ref.num_params(non_embedding=True); del ref

SPEC = {spec}
if name in SPEC:
    d, i, r, L = SPEC[name]
    c = SunnyConfig(vocab_size=vocab, dim=d, inner=i, rank=r, layers=L, h_max=2048.0)
    c.scan_chunk = 64
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
    "name": name, "T": T, "threads": th,
    "stack": model.num_params(non_embedding=True),
    "ms_per_token": round(dt * 1000 / (B * T), 5), "step_ms": round(dt * 1000, 2)}}))
'''


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        name, T, th = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), name=name, T=T, th=th,
                           spec=repr(CFG_SPEC))
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"FAIL {name}@{T}/t{th}: {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["32", "64", "96", "128", "192", "256"])]
    ths = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["8"])]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    names = ["small", "smalln80", "smalln64", "transformer"]

    acc = {(n, T, th): [] for n in names for T in Ts for th in ths}
    for rd in range(rounds):
        for T in Ts:
            for th in ths:
                for n in names:
                    r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                        "--_cell", n, str(T), str(th)],
                                       capture_output=True, text=True, encoding="utf-8")
                    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                    if not line:
                        print(f"  r{rd+1} {n}@{T}/t{th} FAILED")
                        continue
                    acc[(n, T, th)].append(json.loads(line[0]))
                    d = json.loads(line[0])
                    print(f"  r{rd+1} {n:12s} T={T:<5d} th={th} "
                          f"{d['ms_per_token']:.5f} ms/tok stack={d['stack']}")

    out = ROOT / "experiments" / "results" / "_shortseq2.json"
    table = {}
    for th in ths:
        print(f"\n=== 训练态 ms/token 中位 ({rounds} 轮, batch=8, threads={th}) ===")
        print("config".ljust(13) + "".join(f"T={T}".ljust(11) for T in Ts))
        for n in names:
            row = []
            for T in Ts:
                v = statistics.median([d["ms_per_token"] for d in acc[(n, T, th)]]) \
                    if acc[(n, T, th)] else None
                row.append(v)
            table[f"{n}|t{th}"] = row
            cells = "".join(f"{v:.5f}".ljust(11) if v else "n/a".ljust(11) for v in row)
            print(f"  {n:11s}" + cells)
        print("  --- 相对 Transformer 比值 (>1 = 比 TF 慢) ---")
        for n in names[:-1]:
            row = []
            for i, T in enumerate(Ts):
                t = table[f"transformer|t{th}"][i]
                v = table[f"{n}|t{th}"][i]
                row.append(v / t if (v and t) else None)
            cells = "".join(f"{v:.3f}".ljust(11) if v is not None else "n/a".ljust(11)
                            for v in row)
            print(f"  {n:11s}" + cells)

    out.write_text(json.dumps({"Ts": Ts, "threads": ths, "rounds": rounds,
                               "table": table}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()