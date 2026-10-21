"""七架构逐 token 解码对照: 单步耗时 vs 上下文长度.

口径: 每个 (架构, T) 独立子进程; 先逐 token 走满 T 步建立真实解码状态 (不计时),
再测该长度下单步解码 ms; 单线程 (B=1 小张量下多线程只增同步开销, §22.5)。

SunnyNet 单步 O(1), Transformer O(t); 五个 O(N) 同行状态维度固定, 也应平坦。
本脚本给出七条曲线与逐格排名, 定位 SunnyNet 在解码态的真实位次。

用法: python -X utf8 experiments/_decode_seven.py [Ts] [rounds]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SPEC = {"sunny": (80, 160, 244, 4)}
TARGET = 365844
ARCHS = ["sunny", "transformer", "rwkv", "mamba", "h3", "hyena", "linear_attn"]

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params
from sunnynet.linear_baselines import build_linear_baseline, _make
T, name, thr, reps = {T}, "{name}", {thr}, {reps}
torch.set_num_threads(thr); torch.manual_seed(1234)
V = 8192
ref = SunnyNet(SunnyConfig.small(V))
target = ref.num_params(non_embedding=True); del ref
if name in {spec}:
    d, i, r, L = {spec}[name]
    c = SunnyConfig(vocab_size=V, dim=d, inner=i, rank=r, layers=L, h_max=2048.0)
    c.scan_chunk = 64
    m = SunnyNet(c); stt = m.init_state(1)
elif name == "transformer":
    c, _, _ = solve_baseline_for_params(target, V, n_layers=6, n_heads=8, max_len=T + 8)
    m = TransformerLM(c); stt = None
else:
    c, _, _ = build_linear_baseline(name, target, V, max_len=T + 8, n_layers=6)
    m = _make(name, c); stt = m.init_state(1)
m.eval()
idx = torch.randint(0, V, (1, T))
with torch.no_grad():
    for t in range(T):
        _, stt = m.forward_step(idx[:, t], stt)
one = torch.randint(0, V, (1,))
with torch.no_grad():
    m.forward_step(one, stt)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); m.forward_step(one, stt); ts.append(time.perf_counter()-t0)
print("CELL " + json.dumps({{"ms": st.median(ts) * 1000}}))
'''


def reps_for(T: int) -> int:
    return 64 if T <= 1024 else (32 if T <= 2048 else 16)


def cell(name: str, T: int, thr: int = 1) -> float:
    code = CELL.format(root=str(ROOT), T=T, name=name, thr=thr,
                       reps=reps_for(T), spec=repr(SPEC))
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(ROOT), env=dict(os.environ))
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])["ms"]
    raise RuntimeError(f"{name}@{T}: {(r.stderr or r.stdout or '')[-1000:]}")


Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["256", "512", "1024", "2048"])]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
names = ARCHS
print(f"逐 token 解码 ms/step (1 线程, {rounds} 轮)")
acc = {(n, T): [] for n in names for T in Ts}
for rd in range(rounds):
    for T in Ts:
        for n in names:
            v = cell(n, T)
            acc[(n, T)].append(v)
            print(f"  r{rd+1} {n:13s} T={T:<6d} {v:.4f}")

table = {n: [st.median(acc[(n, T)]) for T in Ts] for n in names}
print("\n=== 单步解码 ms 中位 ===")
print("config".ljust(15) + "".join(f"T={T}".ljust(11) for T in Ts))
for n in names:
    print(f"  {n:13s}" + "".join(f"{v:.4f}".ljust(11) for v in table[n]))

print("\n=== 逐格排名 (1=最快) ===")
print("T".ljust(8) + "".join(f"T={T}".ljust(14) for T in Ts))
for n in names:
    row = []
    for i, T in enumerate(Ts):
        rank = sorted(names, key=lambda x: table[x][i]).index(n) + 1
        row.append(f"{rank}/{len(names)}")
    print(f"  {n:8s}" + "".join(r.ljust(14) for r in row))

print("\n=== 相对 SunnyNet 比值 (>1 = 比 SunnyNet 慢) ===")
for n in names:
    if n == "sunny":
        continue
    print(f"  {n:13s}" + "".join(f"{table[n][i]/table['sunny'][i]:.3f}".ljust(11) for i in range(len(Ts))))

out = ROOT / "experiments" / "results" / "_decode_seven.json"
out.write_text(json.dumps({"Ts": Ts, "table": table, "rounds": rounds}, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"\n写出 {out}")