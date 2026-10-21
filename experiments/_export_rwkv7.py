"""把 RWKV-7 GGUF 权重导出为 C Runtime (runtime/rwkv7.c) 的打包 fp32 blob.

布局 (前置 10 个 int32 + 1 个 float + L*4 个 int32 秩表):
  header: magic, version, layers, dim, head_size, n_head, ff, g_rank, vocab, pad
  eps (float)
  ranks: L * (Rw, Ra, Rv, Rg)
  per layer (顺序固定, 见 sn_rk_load):
    ln1 (D) ln1.b (D) ln2 (D) ln2.b (D) lerp (6D)
    r,k,v,o 各存 M^T (D,D)
    w1,w2  (D,Rw) (Rw,D)   a1,a2  (D,Ra) (Ra,D)
    v1,v2  (D,Rv) (Rv,D)   g1,g2  (D,Rg) (Rg,D)
    w0,a0,v0, kk,ka,rk, lnx_w,lnx_b (各 D)
    cmk (D,F) cmv (F,D) cmk_lerp (D)
  尾部: en (D) en.b (D) on (D) on.b (D), embed (vocab,D), head^T (D,vocab)

矩阵存法: 逻辑矩阵 M 满足 y = M @ x, 存 M^T (行主序 (len(x), len(y)))。
  - 非 LoRA 矩阵 (r/k/v/o/cmk/cmv): GGUF 已是 pth 逻辑方向 M, 存 M^T = t_of(pth)。
  - LoRA 矩阵: pth 方向本身即 M^T (numpy 里 `x @ W1`, W1 已是 (in,out)), 故原样存。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.gguf import GGUFModel  # noqa: E402

MAGIC = 0x4B57524B
VERSION = 2

# GGUF 把 LoRA 分解矩阵存成 (out,in), 与官方 .pth 相反.
_LORA = {"time_mix_w1", "time_mix_w2", "time_mix_a1", "time_mix_a2",
         "time_mix_g1", "time_mix_g2", "time_mix_v1", "time_mix_v2"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "experiments/gguf/rwkv7-191M-world-bf16.gguf"
    dst = sys.argv[2] if len(sys.argv) > 2 else "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
    g = GGUFModel(src)

    L = g.get_int("rwkv7.block_count", 12)
    D = g.get_int("rwkv7.embedding_length", 768)
    HS = g.get_int("rwkv7.wkv.head_size", 64)
    NH = D // HS
    F = g.get_int("rwkv7.feed_forward_length", 3072)
    vocab = len(g.tokens())
    eps = g.get_float("rwkv7.attention.layer_norm_epsilon", 1e-5)

    def pth(name: str) -> np.ndarray:
        """pth 逻辑方向的张量; LoRA 矩阵 GGUF 存反, 需转置."""
        a = g.tensor(name)
        if name.split(".")[-2] in _LORA:
            return np.ascontiguousarray(a.T, dtype=np.float32)
        return np.ascontiguousarray(a, dtype=np.float32)

    def t_of(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)

    ranks = []
    for i in range(L):
        p = f"blk.{i}."
        ranks.append((pth(p + "time_mix_w1.weight").shape[1],
                      pth(p + "time_mix_a1.weight").shape[1],
                      pth(p + "time_mix_v1.weight").shape[1],
                      pth(p + "time_mix_g1.weight").shape[1]))

    parts: list[bytes] = []
    parts.append(struct.pack("<10i", MAGIC, VERSION, L, D, HS, NH, F,
                             max(r[3] for r in ranks), vocab, 0))
    parts.append(struct.pack("<f", eps))
    for rw, ra, rv, rg in ranks:
        parts.append(struct.pack("<4i", rw, ra, rv, rg))

    def put(a: np.ndarray):
        parts.append(np.ascontiguousarray(a, dtype=np.float32).reshape(-1).tobytes())

    for i in range(L):
        p = f"blk.{i}."
        rw, ra, rv, rg = ranks[i]
        put(g.tensor(p + "attn_norm.weight"))
        put(g.tensor(p + "attn_norm.bias"))
        put(g.tensor(p + "attn_norm_2.weight"))
        put(g.tensor(p + "attn_norm_2.bias"))
        put(g.tensor(p + "time_mix_lerp_fused.weight").reshape(6, D))

        put(t_of(pth(p + "time_mix_receptance.weight")))
        put(t_of(pth(p + "time_mix_key.weight")))
        put(t_of(pth(p + "time_mix_value.weight")))
        put(t_of(pth(p + "time_mix_output.weight")))

        put(pth(p + "time_mix_w1.weight"))
        put(pth(p + "time_mix_w2.weight"))
        put(pth(p + "time_mix_a1.weight"))
        put(pth(p + "time_mix_a2.weight"))
        put(pth(p + "time_mix_v1.weight"))
        put(pth(p + "time_mix_v2.weight"))
        put(pth(p + "time_mix_g1.weight"))
        put(pth(p + "time_mix_g2.weight"))

        put(g.tensor(p + "time_mix_w0.weight"))
        put(g.tensor(p + "time_mix_a0.weight"))
        put(g.tensor(p + "time_mix_v0.weight"))
        put(g.tensor(p + "time_mix_k_k.weight"))
        put(g.tensor(p + "time_mix_k_a.weight"))
        put(g.tensor(p + "time_mix_r_k.weight"))
        put(g.tensor(p + "time_mix_ln.weight"))
        put(g.tensor(p + "time_mix_ln.bias"))

        put(t_of(pth(p + "channel_mix_key.weight")))
        put(t_of(pth(p + "channel_mix_value.weight")))
        put(g.tensor(p + "channel_mix_lerp_k.weight"))

    put(g.tensor("token_embd_norm.weight"))
    put(g.tensor("token_embd_norm.bias"))
    put(g.tensor("output_norm.weight"))
    put(g.tensor("output_norm.bias"))
    put(g.tensor("token_embd.weight"))
    head = "output.weight" if g.has("output.weight") else "token_embd.weight"
    put(t_of(g.tensor(head)))

    buf = b"".join(parts)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(buf)
    info = {"file": dst, "bytes": len(buf), "layers": L, "dim": D,
            "head_size": HS, "n_head": NH, "ff": F, "vocab": vocab,
            "eps": eps, "ranks_w_a_v_g": ranks[0],
            "rank_v_by_layer": [r[2] for r in ranks],
            "mb": round(len(buf) / 2**20, 1)}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())