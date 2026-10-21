"""RWKV-7 解码速度: numpy vs C 内核 (单线程/多线程), 多轮交错独立子进程.

每格独立子进程. 每轮内三种实现交错, 逐轮配对比较。
只测单步解码 (已建状态), 不含 prefill。
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
BLOB = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
OUT = ROOT / "experiments/results/_rwkv7_c_bench.json"

WORKER = r'''
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(r"{root}")))
which = sys.argv[1]
GGUF = Path(r"{gguf}")
BLOB = Path(r"{blob}")
WARM = "Hello world, this is a warmup sequence for timing."

if which == "numpy":
    from sunnynet.runtime.rwkv7 import RWKV7
    m = RWKV7(GGUF)
    ids = m.tok.encode(WARM)
    st = m.reset(m.init_state())
    for t in ids[:-1]:
        m.step(t, st)
    tok = ids[-1]
    ts = []
    for _ in range(60):
        t0 = time.perf_counter(); m.step(tok, st); ts.append(time.perf_counter() - t0)
else:
    from sunnynet.runtime.rwkv7 import RWKV7
    from sunnynet.runtime.rwkv7_c import SnRWKV7
    ref = RWKV7(GGUF)
    ids = ref.tok.encode(WARM)
    m = SnRWKV7(BLOB, ref.vocab)
    m.reset()
    for t in ids[:-1]:
        m.step_raw(t)
    tok = ids[-1]
    ts = []
    for _ in range(60):
        t0 = time.perf_counter(); m.step_raw(tok); ts.append(time.perf_counter() - t0)

ts.sort()
print(json.dumps({{"which": which, "median_ms": ts[len(ts)//2]*1e3,
                  "min_ms": ts[0]*1e3, "mean_ms": sum(ts)/len(ts)*1e3}}))
'''


def run(which: str, threads: int) -> dict:
    code = WORKER.format(root=str(ROOT), gguf=str(GGUF), blob=str(BLOB))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, which],
                       capture_output=True, text=True, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


CELLS = [("numpy", 1), ("c", 1), ("c", 4), ("c", 8)]


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    acc = {f"{w}-{t}": [] for w, t in CELLS}
    for i in range(rounds):
        line = []
        for w, t in CELLS:
            r = run(w, t)
            acc[f"{w}-{t}"].append(r["median_ms"])
            line.append(f"{w}/{t}T {r['median_ms']:.2f}")
        print(f"  round {i}: " + "  ".join(line))

    med = {k: sorted(v)[rounds // 2] for k, v in acc.items()}
    c1_wins = sum(1 for a, b in zip(acc["c-1"], acc["numpy-1"]) if a < b)
    c8_wins = sum(1 for a, b in zip(acc["c-8"], acc["c-1"]) if a < b)
    summary = {
        "impl": "numpy 参考 vs C 内核 (runtime/rwkv7.c)",
        "note": "单序列, 已建状态后的单步解码; 每格独立子进程; 60 步取中位",
        "rounds": rounds,
        "median_ms": med,
        "speedup_c1_vs_numpy": med["numpy-1"] / med["c-1"],
        "speedup_c8_vs_c1": med["c-1"] / med["c-8"],
        "c1_wins_vs_numpy": f"{c1_wins}/{rounds}",
        "c8_wins_vs_c1": f"{c8_wins}/{rounds}",
        "all_ms": acc,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmedian ms/step: numpy {med['numpy-1']:.2f}  C-1T {med['c-1']:.2f}  "
          f"C-4T {med['c-4']:.2f}  C-8T {med['c-8']:.2f}")
    print(f"C-1T vs numpy: {med['numpy-1']/med['c-1']:.2f}x ({c1_wins}/{rounds})   "
          f"C-8T vs C-1T: {med['c-1']/med['c-8']:.2f}x ({c8_wins}/{rounds})")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())