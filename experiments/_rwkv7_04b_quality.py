"""0.4B 生成质量: INT4 与 F32 同 prompt 贪心对照.

词表从同系列 191M GGUF 提取 (同为 world 65536 词表)。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.gguf import GGUFModel  # noqa: E402
from sunnynet.runtime.rwkv7 import _Tok  # noqa: E402

LIB = ctypes.CDLL(str(ROOT / "sunnynet" / "runtime" / "_build" / "sunnynet_rwkv7.dll"))
V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
LIB.sn_rk_load.restype = V
LIB.sn_rk_load.argtypes = [P]
LIB.sn_rk_step.argtypes = [V, I, V]
LIB.sn_rk_reset.argtypes = [V]
LIB.sn_rk_vocab.restype = I
LIB.sn_rk_vocab.argtypes = [V]


def gen(blob: str, ids: list[int], n: int):
    h = LIB.sn_rk_load(blob.encode())
    vocab = LIB.sn_rk_vocab(h)
    lg = (ctypes.c_float * vocab)()
    LIB.sn_rk_reset(h)
    for t in ids:
        LIB.sn_rk_step(h, int(t), ctypes.byref(lg))
    out = list(ids)
    logits = np.frombuffer(bytes(lg), dtype=np.float32)
    for _ in range(n):
        nxt = int(logits.argmax())
        out.append(nxt)
        LIB.sn_rk_step(h, nxt, ctypes.byref(lg))
        logits = np.frombuffer(bytes(lg), dtype=np.float32)
    return out


def main():
    tk = _Tok(GGUFModel(str(ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf")).tokens())
    prompt = "The Eiffel Tower is located in the city of"
    ids = tk.encode(prompt)
    print("prompt:", prompt)
    print("ids:", ids)
    for tag, blob in [
        ("INT4", r"F:/sunnyclaw-data/models/rwkv7-0.4B-world/rwkv7-0.4b.q4.bin"),
        ("F32 ", r"F:/sunnyclaw-data/models/rwkv7-0.4B-world/rwkv7-0.4b.f32.bin"),
    ]:
        out = gen(blob, ids, 40)
        print(f"[{tag}] {tk.decode(out)}")


if __name__ == "__main__":
    main()