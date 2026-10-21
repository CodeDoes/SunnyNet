"""INT4 量化 GEMV 的精度与速度基准 (AVX2/FMA).

对真实 RWKV-7 的矩阵尺寸, 比较 fp32 GEMV 与 INT4 量化 GEMV。
解码单步是内存带宽受限 (读满全部权重), INT4 把读取量压到 1/8。
每格独立子进程, 多轮取中位。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments/results/_quant_i4_bench.json"

WORKER = r'''
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(r"{root}")))
from sunnynet.runtime import quant as Q

out, in_dim = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(7)
W = (rng.standard_normal((out, in_dim)) * 0.05).astype(np.float32)
x = rng.standard_normal(in_dim).astype(np.float32)

Wq, sq = Q.quant_i4(W, 64)
Wd = Q.dequant_i4(Wq, sq, in_dim, 64)

y_f = Q.gemv_f32(x, W)
y_q = Q.gemv_i4(x, Wq, sq, 64)
y_ref = Q.gemv_f32(x, Wd)
rel = float(np.linalg.norm(y_q - y_f) / np.linalg.norm(y_f))
rel_scalar = float(np.linalg.norm(y_ref - y_f) / np.linalg.norm(y_f))

def timeit(fn, reps=30):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2] * 1e3

t_f = timeit(lambda: Q.gemv_f32(x, W))
t_q = timeit(lambda: Q.gemv_i4(x, Wq, sq, 64))
print(json.dumps({{"out": out, "in": in_dim, "f32_ms": t_f, "i4_ms": t_q,
                  "rel_err_i4": rel, "rel_err_quant_only": rel_scalar,
                  "weight_bytes_f32": out*in_dim*4,
                  "weight_bytes_i4": out*in_dim//2 + out*(in_dim//64)*4}}))
'''


SIZES = [(768, 768), (3072, 768), (768, 3072), (65536, 768)]


def run(out, in_dim, rounds=3):
    code = WORKER.format(root=str(ROOT))
    vals = []
    for _ in range(rounds):
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(out), str(in_dim)],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1200:])
        vals.append(json.loads(r.stdout.strip().splitlines()[-1]))
    med = lambda k: sorted(v[k] for v in vals)[len(vals)//2]
    return {k: med(k) for k in vals[0]}


def main():
    rows = []
    for out, in_dim in SIZES:
        r = run(out, in_dim)
        rows.append(r)
        print(f"  {out}x{in_dim}: f32 {r['f32_ms']:.3f} ms  i4 {r['i4_ms']:.3f} ms  "
              f"speedup {r['f32_ms']/r['i4_ms']:.2f}x  rel_err {r['rel_err_i4']:.3f}")
    summary = {
        "note": "单线程 GEMV 中位; 每格独立子进程, 3 轮取中位; block=64",
        "rows": rows,
        "total_f32_ms": sum(r["f32_ms"] for r in rows),
        "total_i4_ms": sum(r["i4_ms"] for r in rows),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合计 f32 {summary['total_f32_ms']:.2f} ms vs i4 {summary['total_i4_ms']:.2f} ms "
          f"-> {summary['total_f32_ms']/summary['total_i4_ms']:.2f}x")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())