"""解码态内核最终对照: 公平单线程口径 + delta 缓存, 5 轮交错配对.

用法: python -X utf8 experiments/_decode_final.py [T] [rounds]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SPEC = [("n80", 80, 160, 244, 4), ("d160", 160, 320, 200, 6), ("d320", 320, 640, 320, 8)]

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ["SUNNY_KERNEL"] = "{flag}"
import torch
from sunnynet.model import SunnyConfig, SunnyNet
import sunnynet.kernels.core_auto as ca
ca.set_kernel_active({pyflag}, threads=1)
torch.set_num_threads(1); torch.manual_seed(1234)
T, V = {T}, 8192
d, i, r, L = {spec}
c = SunnyConfig(vocab_size=V, dim=d, inner=i, rank=r, layers=L, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).eval()
idx = torch.randint(0, V, (1, T))
def run():
    stt = m.init_state(1)
    with torch.no_grad():
        for t in range(T):
            lg, stt = m.forward_step(idx[:, t], stt)
    return lg
for _ in range(2): run()
ts = []
for _ in range(3):
    t0 = time.perf_counter(); run(); ts.append((time.perf_counter()-t0)/T*1000)
print("CELL " + json.dumps({{"ms": sorted(ts)[1]}}))
'''


def cell(flag, T, spec):
    code = CELL.format(root=str(ROOT), flag="1" if flag else "0",
                       pyflag=str(flag), T=T, spec=repr(spec))
    env = dict(os.environ); env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:/devtools/w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])["ms"]
    raise RuntimeError(f"{flag} {spec}: {r.stderr[-500:]}")


T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
print(f"解码态 B=1 单线程, T={T}, {rounds} 轮交错")
res = {}
for label, d, i, r, L in SPEC:
    p = {0: [], 1: []}
    for rd in range(rounds):
        for f in ([0, 1] if rd % 2 == 0 else [1, 0]):
            p[f].append(cell(bool(f), T, (d, i, r, L)))
    tm, km = st.median(p[0]), st.median(p[1])
    wins = sum(1 for a, b in zip(p[1], p[0]) if a < b)
    ratios = [a/b for a, b in zip(p[1], p[0])]
    res[label] = {"dim": d, "torch": tm, "kernel": km, "ratio": km/tm,
                  "paired_median": st.median(ratios), "wins": wins, "rounds": rounds}
    print(f"  {label:5s} dim={d:4d}  torch {tm:.4f} | kernel {km:.4f} | "
          f"比值 {km/tm:.4f} 配对比值 {st.median(ratios):.4f} 占优 {wins}/{rounds}")
out = ROOT / "experiments" / "results" / "_decode_final.json"
out.write_text(json.dumps({"T": T, "rounds": rounds, "res": res}, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"写出 {out}")
