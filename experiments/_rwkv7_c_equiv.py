"""C 内核 (runtime/rwkv7.c) vs numpy 参考 (runtime/rwkv7.py) 的逐 token 等价.

两者都从同一份 GGUF 导出/读取, 因此这里验证的是 C 实现本身的正确性。
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
BLOB = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
OUT = ROOT / "experiments/results/_rwkv7_c_equiv.json"


def main():
    ref = RWKV7(GGUF)
    c = SnRWKV7(BLOB, ref.vocab)

    text = "\nIn a shocking finding, scientist discovered a herd of dragons"
    ids = ref.tok.encode(text)
    print(f"prompt tokens: {len(ids)}")

    rs = ref.reset(ref.init_state())
    c.reset()
    rows = []
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for i, t in enumerate(ids[:n]):
        rl = ref.step(t, rs)
        cl = np.array(c.step(t), dtype=np.float32)
        d = float(np.abs(rl - cl).max())
        am_r, am_c = int(np.argmax(rl)), int(np.argmax(cl))
        rows.append({"step": i, "max_abs_diff": d,
                     "argmax_ref": am_r, "argmax_c": am_c, "agree": am_r == am_c})
        print(f"  step {i:2d}: max|diff| {d:.3e}  argmax {am_r}/{am_c} "
              f"{'OK' if am_r == am_c else 'MISMATCH'}")

    agree = sum(1 for r in rows if r["agree"])
    summary = {
        "prompt_tokens": len(ids),
        "steps": len(rows),
        "max_abs_diff": max(r["max_abs_diff"] for r in rows),
        "argmax_agree": f"{agree}/{len(rows)}",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmax|diff| overall {summary['max_abs_diff']:.3e}  argmax {summary['argmax_agree']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())