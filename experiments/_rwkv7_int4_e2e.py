"""INT4 量化对 RWKV-7 整模型解码的影响: 速度 / 体积 / 质量.

每格独立子进程, 多轮交错。速度与体积是主要指标; 质量用贪心生成做定性对照。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
FP32 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
INT4 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.q4.bin"
OUT = ROOT / "experiments/results/_rwkv7_int4_e2e.json"

WORKER = r'''
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(r"{root}")))
from sunnynet.runtime.rwkv7 import RWKV7
from sunnynet.runtime.rwkv7_c import SnRWKV7
blob = Path(sys.argv[1])
ref = RWKV7(Path(r"{gguf}"))
m = SnRWKV7(blob, ref.vocab)
m.reset()
ids = ref.tok.encode("Hello world, this is a warmup sequence for timing.")
for t in ids[:-1]:
    m.step_raw(t)
tok = ids[-1]
ts = []
for _ in range(50):
    t0 = time.perf_counter(); m.step_raw(tok); ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({{"median_ms": ts[len(ts)//2]*1e3, "min_ms": ts[0]*1e3}}))
'''


def run(blob: str, threads: int) -> dict:
    code = WORKER.format(root=str(ROOT), gguf=str(GGUF))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(blob)],
                       capture_output=True, text=True, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fp = []
    q4 = []
    for i in range(rounds):
        a = run(FP32, 1)
        b = run(INT4, 1)
        fp.append(a["median_ms"])
        q4.append(b["median_ms"])
        print(f"  round {i}: fp32 {a['median_ms']:.2f} ms  int4 {b['median_ms']:.2f} ms  "
              f"ratio {a['median_ms']/b['median_ms']:.2f}x")

    mfp = sorted(fp)[rounds // 2]
    mq4 = sorted(q4)[rounds // 2]
    wins = sum(1 for a, b in zip(q4, fp) if a < b)
    summary = {
        "note": "RWKV-7 191M 整模型单步, 单序列单线程, 每格独立子进程, 50 步中位",
        "rounds": rounds,
        "fp32_median_ms": mfp,
        "int4_median_ms": mq4,
        "speedup": mfp / mq4,
        "int4_wins": f"{wins}/{rounds}",
        "fp32_blob_mb": round(FP32.stat().st_size / 2**20, 1),
        "int4_blob_mb": round(INT4.stat().st_size / 2**20, 1),
        "all_ms": {"fp32": fp, "int4": q4},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfp32 {mfp:.2f} ms vs int4 {mq4:.2f} ms -> {mfp/mq4:.2f}x ({wins}/{rounds})")
    print(f"blob {summary['fp32_blob_mb']} MB -> {summary['int4_blob_mb']} MB")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())