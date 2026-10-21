"""GGUF 驱动的 RWKV-7 生成与解码耗时基准.

现状: sunnynet/runtime/rwkv7.py 是 numpy 参考实现, 用于验证"统一 Runtime 能读
外部 GGUF 权重并跑出正确结果". 耗时是 numpy 路径, 非 C 内核路径, 如实标注.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.rwkv7 import RWKV7  # noqa: E402

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
OUT = ROOT / "experiments/results/_rwkv7_gguf_bench.json"

PROMPTS = [
    ("en", "The Eiffel Tower is located in the city of"),
    ("en", "Water is composed of hydrogen and"),
    ("zh", "中国的首都是"),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
]


def time_decode(model, n_steps=60):
    """先走满 32 步建状态, 再测 n_steps 次单步中位数."""
    state = model.reset(model.init_state())
    warm = model.tok.encode("Hello world, this is a warmup sequence for timing.")
    for t in warm:
        model.step(t, state)
    ts = []
    tok = warm[-1]
    for _ in range(n_steps):
        t0 = time.perf_counter()
        model.step(tok, state)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {
        "median_ms": ts[len(ts) // 2] * 1e3,
        "min_ms": ts[0] * 1e3,
        "mean_ms": sum(ts) / len(ts) * 1e3,
    }


def main():
    t0 = time.perf_counter()
    model = RWKV7(GGUF)
    load_s = time.perf_counter() - t0

    samples = []
    for tag, p in PROMPTS:
        text, ids = model.generate(p, max_new=24, temperature=0.7, top_k=20, seed=42)
        samples.append({"tag": tag, "prompt": p, "output": text})
        print(f"[{tag}] {text!r}")

    timing = time_decode(model)
    summary = {
        "model": "rwkv7-191M-world-bf16",
        "params_label": "191M",
        "n_layer": model.n_layer,
        "n_embd": model.n_embd,
        "head_size": model.head_size,
        "vocab": model.vocab,
        "load_seconds": load_s,
        "decode_ms_median": timing["median_ms"],
        "decode_ms_min": timing["min_ms"],
        "decode_ms_mean": timing["mean_ms"],
        "engine": "numpy 参考路径 (非 C 内核)",
        "note": "单序列单线程; 状态每步 O(1) 与上下文无关",
        "samples": samples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nload {load_s:.2f}s  decode median {timing['median_ms']:.2f} ms/step "
          f"(min {timing['min_ms']:.2f}, mean {timing['mean_ms']:.2f})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()