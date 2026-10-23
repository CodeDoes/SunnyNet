"""0.4B RWKV-7 单步解码实测: 每格独立子进程, 多轮交错.

口径同 _rwkv7_c_bench.py: 只测单步 (已建状态), 不含 prefill。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"

WORKER = r'''
import json, sys, time, ctypes, os
blob = r"{blob}"
threads = int("{threads}")
os.environ["OMP_NUM_THREADS"] = str(threads)
LIB = ctypes.CDLL(r"{lib}")
V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
LIB.sn_rk_load.restype = V
LIB.sn_rk_load.argtypes = [P]
LIB.sn_rk_step.argtypes = [V, I, V]
LIB.sn_rk_reset.argtypes = [V]
LIB.sn_rk_vocab.restype = I
LIB.sn_rk_vocab.argtypes = [V]
h = LIB.sn_rk_load(blob.encode())
if not h:
    print(json.dumps({{"error": "load failed"}})); sys.exit(1)
vocab = LIB.sn_rk_vocab(h)
lg = (ctypes.c_float * vocab)()
LIB.sn_rk_reset(h)
for t in range(1, 40):
    LIB.sn_rk_step(h, t, ctypes.byref(lg))
ts = []
for i in range(80):
    t0 = time.perf_counter()
    LIB.sn_rk_step(h, 40 + i, ctypes.byref(lg))
    ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({{"median_ms": ts[len(ts)//2]*1e3, "min_ms": ts[0]*1e3,
                  "mean_ms": sum(ts)/len(ts)*1e3, "n": len(ts)}}))
'''


def run(blob: str, threads: int) -> dict:
    code = WORKER.format(
        blob=blob, threads=threads,
        lib=str(ROOT / "sunnynet" / "runtime" / "_build" / "sunnynet_rwkv7.dll"))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    blob = sys.argv[1] if len(sys.argv) > 1 else \
        r"F:/sunnyclaw-data/models/rwkv7-0.4B-world/rwkv7-0.4b.q4.bin"
    rounds = 5
    res = {}
    for th in (1, 4, 6, 8):
        res[th] = []
    for r in range(rounds):
        for th in (1, 4, 6, 8):
            res[th].append(run(blob, th))
    print(f"{'线程':>4} {'中位 ms':>10} {'最小 ms':>10} {'均值 ms':>10}")
    summary = {}
    for th, rs in res.items():
        med = sorted(x["median_ms"] for x in rs)
        mn = sorted(x["min_ms"] for x in rs)
        print(f"{th:>4} {med[len(med)//2]:>10.2f} {mn[0]:>10.2f} "
              f"{sum(x['mean_ms'] for x in rs)/len(rs):>10.2f}")
        summary[str(th)] = [x["median_ms"] for x in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "f32" if "f32" in blob else "q4"
    dst = OUT / f"_rwkv7_04b_{tag}_bench.json"
    dst.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()