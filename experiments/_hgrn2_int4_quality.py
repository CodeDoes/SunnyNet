"""HGRN2 fp32 vs INT4 的真实文本贪心续写对照.

合成 token 序列只能看数值; 质量要看真实 prompt 的续写是否连贯。
用法: python -X utf8 experiments/_hgrn2_int4_quality.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_HGRN2",
                      str(Path("F:/sunnyclaw-data/models/hgrn2-1.3B-100B")))
BLOB32 = Path(MDIR) / "hgrn2.c.bin"
BLOB4 = Path(MDIR) / "hgrn2.q4.bin"

PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The theory of relativity was proposed by",
    "The sun rises in the east and sets in the",
]


def main():
    import numpy as np
    from transformers import AutoTokenizer
    from sunnynet.runtime.hgrn2_c import SnHGRN2

    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    vocab = tok.vocab_size
    h32 = SnHGRN2(BLOB32, vocab)
    h4 = SnHGRN2(BLOB4, vocab)

    def gen(h, ids, n=20):
        h.reset()
        for t in ids[:-1]:
            h.step_raw(int(t))
        out = []
        nxt = int(ids[-1])
        for _ in range(n):
            logits = np.asarray(h.step(nxt))
            nxt = int(logits.argmax())
            out.append(nxt)
        return out

    for p in PROMPTS:
        ids = tok(p, return_tensors=None)["input_ids"]
        a = gen(h32, ids)
        b = gen(h4, ids)
        eq = sum(int(x == y) for x, y in zip(a, b))
        print(f"\nprompt: {p!r}")
        print(f"  fp32 : {p}{tok.decode(a)!r}")
        print(f"  int4 : {p}{tok.decode(b)!r}")
        print(f"  逐 token 一致: {eq}/{len(a)}")


if __name__ == "__main__":
    main()