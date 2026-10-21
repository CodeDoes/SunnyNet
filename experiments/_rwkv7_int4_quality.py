"""INT4 量化对 RWKV-7 生成质量的影响对照.

口径: 同一 prompt, 贪心解码对比 INT4 / fp32 的 token 轨迹与 logits 偏差;
再用少量 prompt 的人工可读性做定性判断。质量不是 ppl — RWKV-7 无本项目验证集。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.rwkv7 import RWKV7  # noqa: E402
from sunnynet.runtime.rwkv7_c import SnRWKV7  # noqa: E402

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
FP32 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
INT4 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.q4.bin"
OUT = ROOT / "experiments/results/_rwkv7_int4_quality.json"

PROMPTS = [
    ("en_fact", "The Eiffel Tower is located in the city of"),
    ("en_math", "Water is composed of hydrogen and"),
    ("zh_fact", "中国的首都是"),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
]


def greedy(model, ref, prompt, n=24):
    ids = ref.tok.encode(prompt)
    model.reset()
    lg = np.array(model.prefill(ids), dtype=np.float32)
    out = list(ids)
    logit_rows = []
    for _ in range(n):
        logit_rows.append(lg)
        nxt = int(np.argmax(lg))
        out.append(nxt)
        lg = np.array(model.step(nxt), dtype=np.float32)
    return out, np.array(logit_rows, dtype=np.float32)


def main():
    ref = RWKV7(GGUF)
    fp = SnRWKV7(FP32, ref.vocab)
    q4 = SnRWKV7(INT4, ref.vocab)

    rows = []
    for tag, p in PROMPTS:
        o_f, l_f = greedy(fp, ref, p)
        o_q, l_q = greedy(q4, ref, p)
        d = float(np.abs(l_f - l_q).max())
        # 前 8 步 argmax 一致率 (之后轨迹分叉是量化的正常后果)
        agree = sum(int(a == b) for a, b in zip(o_f, o_q))
        rows.append({
            "tag": tag,
            "prompt": p,
            "logits_max_abs_diff": d,
            "argmax_agree_over_len": f"{agree}/{len(o_f)}",
            "fp32_text": ref.tok.decode(o_f),
            "int4_text": ref.tok.decode(o_q),
        })
        print(f"[{tag}] logits max|diff| {d:.3f}  argmax 一致 {agree}/{len(o_f)}")
        print(f"   fp32: {rows[-1]['fp32_text']!r}")
        print(f"   int4: {rows[-1]['int4_text']!r}")

    summary = {
        "note": "INT4 vs fp32 同 prompt 贪心对比; 无本项目验证集, 不作 ppl 陈述",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())