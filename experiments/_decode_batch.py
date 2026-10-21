"""解码态内核 vs torch 的 batch 交叉点: B=1 时融合亏在 FFI, B 大时才回本.

用法: python -X utf8 experiments/_decode_batch.py [T] [Bs] [rounds]
"""
from __future__ import annotations
import json, os, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.environ["SUNNY_KERNEL"] = "{flag}"
import torch
from sunnynet.model import SunnyConfig, SunnyNet
import sunnynet.kernels.core_auto as ca
ca.set_kernel_active({pyflag}, threads=8)
torch.set_num_threads(8); torch.manual_seed(1234)
T, B, V = {T}, {B}, 8192
c = SunnyConfig(vocab_size=V, dim=80, inner=160, rank=244, layers=4, h_max=2048.0)
c.scan_chunk = 64
m = SunnyNet(c).eval()
idx = torch.randint(0, V, (B, T))
def run():
    stt = m.init_state(B)
    with torch.no_grad():
        for t in range(T):
            lg, stt = m.forward_step(idx[:, t], stt)
    return lg
for _ in range(2): run()
t0 = time.perf_counter()
for _ in range(3): run()
dt = (time.perf_counter() - t0) / 3
print("CELL " + json.dumps({{"mode": "{mode}", "T": T, "B": B,
      "ms_per_step": round(dt*1000/T, 5)}}))
'''


def cell(mode, T, B):
    code = CELL.format(root=str(ROOT), flag="1" if mode == "kernel" else "0",
                       pyflag=str(mode == "kernel"), T=T, B=B, mode=mode)
    env = dict(os.environ); env["SUNNY_W64"] = env.get("SUNNY_W64", r"F:/devtools/w64devkit")
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=env)
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])
    raise RuntimeError(f"{mode}@B{B}: {r.stderr[-800:]}")


T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
Bs = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "4", "8", "16", "32"])]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
print(f"T={T} 解码态 (B, 3 轮交错)")
res = {}
for B in Bs:
    p = {"torch": [], "kernel": []}
    for r in range(rounds):
        for m_ in (["torch", "kernel"] if r % 2 == 0 else ["kernel", "torch"]):
            p[m_].append(cell(m_, T, B)["ms_per_step"])
    tm, km = st.median(p["torch"]), st.median(p["kernel"])
    wins = sum(1 for a, b in zip(p["kernel"], p["torch"]) if a < b)
    res[B] = {"torch": tm, "kernel": km, "ratio": km / tm, "wins": wins}
    print(f"  B={B:2d}  torch {tm:.5f} | kernel {km:.5f} ms/step | "
          f"kernel/torch={km/tm:.3f} | 占优 {wins}/{rounds}")
out = ROOT / "experiments" / "results" / "_decode_batch.json"
out.write_text(json.dumps({"T": T, "rounds": rounds, "res": {str(k): v for k, v in res.items()}},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写出 {out}")
