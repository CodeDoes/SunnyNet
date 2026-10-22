"""Mamba-2 / DeltaNet 单步解码耗时, 每格独立子进程 + 多轮交错.

口径同 RWKV-7 的 _rwkv7_c_bench.py: 每轮每架构独立进程, 逐轮配对比较。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable

MAMBA = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))
DELTA = os.environ.get("SUNNY_DELTANET", str(ROOT / "models" / "deltanet-pure"))

WORKER = r'''
import sys, time, json
sys.path.insert(0, r"{root}")
import numpy as np
which = "{which}"
steps = {steps}
if which == "mamba2":
    from sunnynet.runtime.mamba2 import Mamba2 as M
    m = M(r"{mamba}")
elif which == "deltanet":
    from sunnynet.runtime.deltanet import DeltaNet as M
    m = M(r"{delta}")
st = m.reset(m.init_state())
ids = list(range(1, 40))
for t in ids:
    m.step(int(t), st)
ts = []
for i in range(steps):
    t0 = time.perf_counter()
    m.step(int(ids[i % len(ids)]), st)
    ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({{"median_ms": ts[len(ts)//2] * 1e3,
                  "mean_ms": sum(ts)/len(ts) * 1e3,
                  "steps": steps}}))
'''


def run(which, steps=30):
    src = WORKER.format(root=str(ROOT), which=which, steps=steps,
                       mamba=MAMBA, delta=DELTA)
    r = subprocess.run([PY, "-X", "utf8", "-c", src], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"{which} 失败:\n{(r.stderr or '')[-1500:]}")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = {"mamba2": [], "deltanet": []}
    for r in range(rounds):
        for which in ("mamba2", "deltanet"):
            try:
                d = run(which)
                res[which].append(d["median_ms"])
            except Exception as e:
                print(f"round{r} {which} 失败: {e}")
        print(f"round {r}: " + "  ".join(
            f"{k}={v[-1]:.2f}ms" if v else f"{k}=NA" for k, v in res.items()))

    print("\n=== 汇总 (ms/token, 单线程 numpy 参考路径) ===")
    for k, v in res.items():
        if v:
            v2 = sorted(v)
            print(f"{k:9s} median={v2[len(v2)//2]:.2f}  "
                  f"min={min(v):.2f}  max={max(v):.2f}  n={len(v)}")
    Path(HERE.parent / "results" / "_hf_runtime_bench.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()