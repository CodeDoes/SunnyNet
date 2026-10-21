"""解码态内核收益 vs 模型宽度: D=80 时 FFI 主导, 更宽的真实预设是否反转.

用法: python -X utf8 experiments/_decode_dim.py [T] [rounds]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# (label, dim, inner, rank, layers)
SPEC = [("n80", 80, 160, 244, 4), ("d160", 160, 320, 200, 6),
        ("d320", 320, 640, 320, 8), ("d512", 512, 896, 320, 10)]

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
print("CELL " + json.dumps({{"mode": "{mode}", "ms_per_step": sorted(ts)[1],
      "stack": m.num_params(non_embedding=True)}}))
'''


def cell(mode, T, spec):
    code = CELL.format(root=str(ROOT), flag="1" if mode == "kernel" else "0",
                       pyflag=str(mode == "kernel"), T=T, spec=repr(spec), mode=mode)
    env = dict(os.environ); env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:/devtools/w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])
    raise RuntimeError(f"{mode} {spec}: {r.stderr[-600:]}")


T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
print(f"T={T} B=1 单线程口径, 解码态 (dim 扫描)")
res = {}
for label, d, i, r, L in SPEC:
    p = {"torch": [], "kernel": []}
    for rd in range(rounds):
        for m_ in (["torch", "kernel"] if rd % 2 == 0 else ["kernel", "torch"]):
            p[m_].append(cell(m_, T, (d, i, r, L))["ms_per_step"])
    tm, km = st.median(p["torch"]), st.median(p["kernel"])
    wins = sum(1 for a, b in zip(p["kernel"], p["torch"]) if a < b)
    res[label] = {"dim": d, "torch": tm, "kernel": km, "ratio": km / tm, "wins": wins,
                  "rounds": rounds}
    print(f"  {label:5s} dim={d:4d}  torch {tm:.4f} | kernel {km:.4f} ms/step | "
          f"kernel/torch={km/tm:.3f} | 占优 {wins}/{rounds}")
out = ROOT / "experiments" / "results" / "_decode_dim.json"
out.write_text(json.dumps({"T": T, "res": res}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写出 {out}")
