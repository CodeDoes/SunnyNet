"""FLA 权重导出的 blob 与 GGUF 路线 blob 的端到端一致性.

两个 blob 来自同一 191M 模型的两种发布格式。映射若正确, 逐 token logits 应一致
(L0 的 v_lora 不被前向使用, 允许占位差异)。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIB = ctypes.CDLL(str(ROOT / "sunnynet" / "runtime" / "_build" / "sunnynet_rwkv7.dll"))
V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
LIB.sn_rk_load.restype = V
LIB.sn_rk_load.argtypes = [P]
LIB.sn_rk_step.argtypes = [V, I, V]
LIB.sn_rk_reset.argtypes = [V]
LIB.sn_rk_vocab.restype = I
LIB.sn_rk_vocab.argtypes = [V]


def run(blob: str, ids: list[int]):
    h = LIB.sn_rk_load(blob.encode())
    if not h:
        raise RuntimeError(f"load 失败: {blob}")
    vocab = LIB.sn_rk_vocab(h)
    lg = (ctypes.c_float * vocab)()
    LIB.sn_rk_reset(h)
    out = []
    for t in ids:
        LIB.sn_rk_step(h, int(t), ctypes.byref(lg))
        out.append(np.frombuffer(bytes(lg), dtype=np.float32).copy())
    return np.array(out), vocab


def main():
    a = sys.argv[1]
    b = sys.argv[2]
    ids = [1] + [100 + i * 7 for i in range(40)]
    la, va = run(a, ids)
    lb, vb = run(b, ids)
    print(f"vocab: {va} vs {vb}")
    d = np.abs(la - lb)
    print(f"logits max|diff| = {d.max():.3e}   mean = {d.mean():.3e}")
    print(f"argmax 一致: {int((la.argmax(-1) == lb.argmax(-1)).sum())}/{len(ids)}")
    rel = d.max() / max(np.abs(lb).max(), 1e-9)
    print(f"相对 max = {rel:.3e}")


if __name__ == "__main__":
    main()