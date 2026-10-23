"""DeltaNet 单步解码: numpy 参考 vs 整模型 C 内核.

口径同 RWKV-7 的 _rwkv7_c_bench.py: 每格独立子进程, 多轮交错, 逐轮配对比较。
单次读数会虚高, 故看逐轮胜负而非只看中位。
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
MDIR = os.environ.get("SUNNY_DELTANET", str(ROOT / "models" / "deltanet-pure"))

WORKER = r'''
import sys, time, json, ctypes
sys.path.insert(0, r"{root}")
import numpy as np
which = "{which}"
steps = {steps}
mdir = r"{mdir}"
if which == "numpy":
    from sunnynet.runtime.deltanet import DeltaNet
    m = DeltaNet(mdir); st = m.reset(m.init_state())
    ids = list(range(1, 40))
    for t in ids: m.step(int(t), st)
    ts = []
    for i in range(steps):
        t0 = time.perf_counter(); m.step(int(ids[i % len(ids)]), st)
        ts.append(time.perf_counter() - t0)
else:
    lib = ctypes.CDLL(r"{root}/sunnynet/runtime/_build/sunnynet_deltanet.dll")
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn_dn_load.restype = V; lib.sn_dn_load.argtypes = [P]
    lib.sn_dn_step.argtypes = [V, I, V]
    lib.sn_dn_reset.argtypes = [V]
    lib.sn_dn_vocab.restype = I; lib.sn_dn_vocab.argtypes = [V]
    h = lib.sn_dn_load((mdir + "/deltanet.c.bin").encode())
    vocab = lib.sn_dn_vocab(h)
    logits = (ctypes.c_float * vocab)()
    ids = list(range(1, 40))
    for t in ids: lib.sn_dn_step(h, int(t), ctypes.byref(logits))
    ts = []
    for i in range(steps):
        t0 = time.perf_counter()
        lib.sn_dn_step(h, int(ids[i % len(ids)]), ctypes.byref(logits))
        ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({{"median_ms": ts[len(ts)//2] * 1e3,
                  "mean_ms": sum(ts)/len(ts) * 1e3, "steps": steps}}))
'''


def run(which, steps=40):
    src = WORKER.format(root=str(ROOT), which=which, steps=steps, mdir=MDIR)
    r = subprocess.run([PY, "-X", "utf8", "-c", src], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"{which} 失败:\n{(r.stderr or '')[-1500:]}")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = {"numpy": [], "c": []}
    win = 0
    for r in range(rounds):
        for which in ("numpy", "c"):
            res[which].append(run(which)["median_ms"])
        if res["c"][-1] < res["numpy"][-1]:
            win += 1
        print(f"round {r}: numpy={res['numpy'][-1]:.2f}ms  "
              f"c={res['c'][-1]:.2f}ms  (c 快 {res['numpy'][-1]/res['c'][-1]:.2f}x)")

    print(f"\n=== 汇总 (ms/token, DeltaNet 340M 单步) ===")
    for k, v in res.items():
        v2 = sorted(v)
        print(f"{k:6s} median={v2[len(v2)//2]:.2f}  min={min(v):.2f}  "
              f"max={max(v):.2f}  n={len(v)}")
    nm = sorted(res["numpy"])[len(res["numpy"]) // 2]
    cm = sorted(res["c"])[len(res["c"]) // 2]
    print(f"C 逐轮占优 {win}/{rounds}  中位倍数 {nm/cm:.2f}x")
    (ROOT / "results" / "_deltanet_c_bench.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()