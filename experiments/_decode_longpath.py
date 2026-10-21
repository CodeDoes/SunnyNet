"""推理出路验证: 长上下文解码下单步耗时是否随上下文增长.

短 T 小张量内核无解 (FFI 与单算子同量级), 但单步解码复杂度是架构级的:
SunnyNet 单步 O(1), Transformer 单步 O(t)。本脚本测 "单步解码耗时 vs 上下文
长度" 这条曲线, 交叉点即长上下文推理的实际胜出点。

口径: 每个 T 独立子进程; 先逐 token 走满 T 步建立真实解码缓存 (走满不计时,
避免 Transformer 被并行 prefill 加速), 再测该长度下单步解码 ms; 1 线程。

用法: python -X utf8 experiments/_decode_longpath.py [Ts] [rounds]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SPEC = {"sunny": (80, 160, 244, 4), "small": (128, 192, 64, 4)}
TARGET = 365844

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params
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
else:
    c, _, _ = solve_baseline_for_params(target, V, n_layers=6, n_heads=8, max_len=T + 8)
    m = TransformerLM(c); stt = None
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
    env = dict(os.environ)
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])["ms"]
    raise RuntimeError(f"{name}@{T}: {r.stderr[-800:]}")


Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["256", "512", "1024", "2048", "4096"])]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
print(f"单步解码 ms (1 线程, {rounds} 轮): SunnyNet 应平坦, Transformer 应 ∝T")
names = ["sunny", "transformer"]
acc = {(n, T): [] for n in names for T in Ts}
for rd in range(rounds):
    for T in Ts:
        for n in names:
            acc[(n, T)].append(cell(n, T))
            print(f"  r{rd+1} {n:12s} T={T:<6d} {acc[(n,T)][-1]:.4f} ms/step")
table = {}
print("\n=== 单步解码 ms 中位 ===")
print("config".ljust(13) + "".join(f"T={T}".ljust(11) for T in Ts))
for n in names:
    row = [st.median(acc[(n, T)]) for T in Ts]
    table[n] = row
    print(f"  {n:11s}" + "".join(f"{v:.4f}".ljust(11) for v in row))
print("\n=== sunny / transformer 单步比值 (<1 = sunny 快) ===")
print("  " + "sunny/tf".ljust(11) + "".join(
    f"{(table['sunny'][i] / table['transformer'][i]):.3f}".ljust(11) for i in range(len(Ts))))
base = table["sunny"][0]
print("\n=== SunnyNet 单步随 T 的漂移 (相对 T=min) ===")
print("  " + "相对".ljust(11) + "".join(f"{(v / base):.3f}".ljust(11) for v in table["sunny"]))
out = ROOT / "experiments" / "results" / "_decode_longpath.json"
out.write_text(json.dumps({"Ts": Ts, "table": table}, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"写出 {out}")