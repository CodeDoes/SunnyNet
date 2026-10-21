"""批处理解码态对照: batch 对逐 token 解码吞吐的影响.

动机: §20.2 测的是 batch=1 解码 (SunnyNet 慢 1.08-2.14x). 但真实服务是批推理,
且训练态 (batch=8) 下 SunnyNet 是快的一方 ---- 批解码可能改变结论.
本节直接测 batch 维对两边的影响, 判定解码态倒挂是否是 batch=1 的伪影.

口径: eval + no_grad, 每步喂 batch 的一列 token, 报 ms/token (总时间/(B*T)).
每格独立子进程, 3 轮交错取中位.

用法: python -X utf8 experiments/_batch_decode.py [Ts] [Bs] [rounds]
      python -X utf8 experiments/_batch_decode.py --_cell <name> <T> <B>
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
    st = model.init_state(B) if name in {spec} else None
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
    "name": name, "T": T, "B": B,
    "ms_per_step": round(dt * 1000 / T, 5),
    "ms_per_token": round(dt * 1000 / (B * T), 5),
    "tok_per_sec": round(B * T / dt, 1)}}))
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
            print(f"FAIL {name}@{T}/B{B}: {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["256", "1024"])]
    Bs = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "8"])]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    names = ["small", "smalln80", "transformer"]

    acc = {(n, T, B): [] for n in names for T in Ts for B in Bs}
    for rd in range(rounds):
        for T in Ts:
            for B in Bs:
                for n in names:
                    r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                        "--_cell", n, str(T), str(B)],
                                       capture_output=True, text=True, encoding="utf-8")
                    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                    if not line:
                        print(f"  r{rd+1} {n}@{T}/B{B} FAILED"); continue
                    acc[(n, T, B)].append(json.loads(line[0]))
                    d = json.loads(line[0])
                    print(f"  r{rd+1} {n:12s} T={T:<5d} B={B:<3d} "
                          f"{d['ms_per_token']:.5f} ms/tok {d['tok_per_sec']} tok/s")

    out = ROOT / "experiments" / "results" / "_batch_decode.json"
    table = {}
    for T in Ts:
        print(f"\n=== 解码态 ms/token 中位 ({rounds} 轮, T={T}) ===")
        print("config".ljust(13) + "".join(f"B={B}".ljust(11) for B in Bs))
        for n in names:
            row = []
            for B in Bs:
                v = statistics.median([d["ms_per_token"] for d in acc[(n, T, B)]]) \
                    if acc[(n, T, B)] else None
                row.append(v)
            table[f"{n}|T{T}"] = row
            print(f"  {n:11s}" + "".join(f"{v:.5f}".ljust(11) if v else "n/a".ljust(11)
                                         for v in row))
        print("  --- 相对 Transformer 比值 (>1 = 比 TF 慢) ---")
        for n in names[:-1]:
            row = []
            for i, B in enumerate(Bs):
                t = table[f"transformer|T{T}"][i]
                v = table[f"{n}|T{T}"][i]
                row.append(v / t if (v and t) else None)
            print(f"  {n:11s}" + "".join(f"{v:.3f}".ljust(11) if v is not None
                                         else "n/a".ljust(11) for v in row))

    out.write_text(json.dumps({"Ts": Ts, "Bs": Bs, "rounds": rounds, "table": table},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()