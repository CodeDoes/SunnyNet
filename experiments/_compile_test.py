"""torch.compile (Inductor) 可行性 + 加速实测.

运行 (需先 vcvars64):
  python -X utf8 experiments/_compile_test.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def bench(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1000, ts[0] * 1000


def main():
    torch.set_num_threads(8)
    vocab, B, T = 8192, 8, 1024
    torch.manual_seed(0)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step(model):
        lg = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print("[1] Eager 基线 ...", flush=True)
    step(m)
    e_ms, e_min = bench(lambda: step(m))

    print("[2] torch.compile 编译中 (首次含编译耗时) ...", flush=True)
    t0 = time.perf_counter()
    try:
        cm = torch.compile(m, backend="inductor", dynamic=False)
        step(cm)
        compile_s = time.perf_counter() - t0
        print(f"    编译成功, 耗时 {compile_s:.1f}s", flush=True)
    except Exception as e:
        print("RESULT " + json.dumps({
            "status": "compile_failed",
            "err": f"{type(e).__name__}: {str(e)[:400]}",
            "eager_ms": round(e_ms, 1),
        }))
        return

    c_ms, c_min = bench(lambda: step(cm), reps=9, warm=2)
    print("RESULT " + json.dumps({
        "status": "ok",
        "eager_ms": round(e_ms, 2), "eager_min": round(e_min, 2),
        "compiled_ms": round(c_ms, 2), "compiled_min": round(c_min, 2),
        "speedup": round(e_ms / c_ms, 3),
        "compile_seconds": round(compile_s, 1),
        "T": T, "B": B,
    }))


if __name__ == "__main__":
    main()