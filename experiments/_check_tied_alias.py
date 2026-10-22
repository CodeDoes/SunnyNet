"""检查 runtime blob 里 embed 与 head_t 是否为同一矩阵。

tie_embeddings=true 时 head_t 的行 j 应等于 embed 的行 j。
若成立, 运行时可以让 head_t 别名到 embed, 每步少读 vocab*dim*4 字节。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"


def main() -> None:
    vocab, dim, inner, rank, taps, layers = 8192, 1408, 1920, 1024, 4, 8
    per_layer = (2 * dim + 2 * inner + taps * dim + dim * rank + rank * 2 * dim
                 + dim * inner + inner * dim + dim * dim + dim + 2)
    n_pre = layers * per_layer + 2 + 2 * dim
    emb_off = 1 + n_pre          # 元素偏移 (跳过 magic 之外的 9 个 int32 已在校验里)
    raw = np.fromfile(BLOB, dtype=np.int32, count=10)
    print("header:", raw.tolist())
    body_off_bytes = 40
    emb_bytes = vocab * dim * 4
    head_bytes = dim * vocab * 4
    with open(BLOB, "rb") as f:
        f.seek(body_off_bytes + n_pre * 4)
        emb = np.frombuffer(f.read(emb_bytes), dtype=np.float32).reshape(vocab, dim)
        head = np.frombuffer(f.read(head_bytes), dtype=np.float32).reshape(dim, vocab)
    print("embed shape", emb.shape, "head_t shape", head.shape)
    diff_tt = float(np.abs(emb - head.T).max())
    diff_flat = float(np.abs(emb.ravel() - head.ravel()).max())
    print(f"max|embed - head_t.T| = {diff_tt:.3e}")
    print(f"max|embed.ravel - head.ravel| = {diff_flat:.3e}")
    print("identical:", diff_tt == 0.0)
    scale = float(np.abs(emb).max())
    print(f"scale max|embed| = {scale:.4f}")


if __name__ == "__main__":
    main()