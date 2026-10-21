"""把 RWKV-7 GGUF 权重导出为 C Runtime 的打包 blob (v3, 支持 INT4).

布局 (前置 10 个 int32 + 1 个 float + L*4 个 int32 秩表, 其后为带标志的张量流):
  header: magic, version=3, layers, dim, head_size, n_head, ff, g_rank, vocab, 0
  eps (float)
  ranks: L * (Rw, Ra, Rv, Rg)
  张量流 (每组前置 1 个 int32 标志: 1=INT4, 0=fp32; INT4 载荷补齐到 4 字节):
    per layer: ln1 ln1.b ln2 ln2.b lerp(6D)
      r,k,v,o 各存 M^T (D,D)
      w1 (D,Rw) w2 (Rw,D) a1 (D,Ra) a2 (Ra,D)
      v1 (D,Rv) v2 (Rv,D) g1 (D,Rg) g2 (Rg,D)
      w0 a0 v0 kk ka rk lnx_w lnx_b (各 D)
      cmk (D,F) cmv (F,D) cmk_lerp (D)
    尾部: en en.b on on.b (各 D), embed (vocab,D, 查表用保持 fp32),
          head^T (D,vocab)

矩阵存法: 逻辑矩阵 M 满足 y = M @ x, 存 M^T (行主序 (len(x), len(y)))。
  INT4 量化 2 维矩阵; 向量与嵌入表保持 fp32 (嵌入是查表不是乘加)。
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
from sunnynet.runtime import quant as Q  # noqa: E402

MAGIC = 0x4B57524B
VERSION = 3
BLOCK = 64

_LORA = {"time_mix_w1", "time_mix_w2", "time_mix_a1", "time_mix_a2",
         "time_mix_g1", "time_mix_g2", "time_mix_v1", "time_mix_v2"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "experiments/gguf/rwkv7-191M-world-bf16.gguf"
    dst = sys.argv[2] if len(sys.argv) > 2 else "experiments/gguf/rwkv7-191M-world-bf16.q4.bin"
    use_int4 = "--f32" not in sys.argv
    g = GGUFModel(src)

    L = g.get_int("rwkv7.block_count", 12)
    D = g.get_int("rwkv7.embedding_length", 768)
    HS = g.get_int("rwkv7.wkv.head_size", 64)
    NH = D // HS
    F = g.get_int("rwkv7.feed_forward_length", 3072)
    vocab = len(g.tokens())
    eps = g.get_float("rwkv7.attention.layer_norm_epsilon", 1e-5)

    def pth(name: str) -> np.ndarray:
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
                             max(r[3] for r in ranks), vocab, 1 if use_int4 else 0))
    parts.append(struct.pack("<f", eps))
    for rw, ra, rv, rg in ranks:
        parts.append(struct.pack("<4i", rw, ra, rv, rg))

    stats = {"int4_mats": 0, "f32_mats": 0}

    def put(a: np.ndarray, quantize: bool = False):
        a = np.ascontiguousarray(a, dtype=np.float32)
        # a 为 (in,out); INT4 需 (out,in)
        can_q = (quantize and use_int4 and a.ndim == 2 and a.shape[0] % BLOCK == 0
                 and a.shape[1] > 0)
        if can_q:
            q, s = Q.quant_i4(np.ascontiguousarray(a.T), BLOCK)
            parts.append(struct.pack("<i", 1))
            parts.append(q.tobytes() + s.astype(np.float32).tobytes())
            stats["int4_mats"] += 1
        else:
            parts.append(struct.pack("<i", 0))
            parts.append(a.reshape(-1).tobytes())
            stats["f32_mats"] += 1

    for i in range(L):
        p = f"blk.{i}."
        put(g.tensor(p + "attn_norm.weight"))
        put(g.tensor(p + "attn_norm.bias"))
        put(g.tensor(p + "attn_norm_2.weight"))
        put(g.tensor(p + "attn_norm_2.bias"))
        put(g.tensor(p + "time_mix_lerp_fused.weight").reshape(6, D))

        put(t_of(pth(p + "time_mix_receptance.weight")), True)
        put(t_of(pth(p + "time_mix_key.weight")), True)
        put(t_of(pth(p + "time_mix_value.weight")), True)
        put(t_of(pth(p + "time_mix_output.weight")), True)

        put(pth(p + "time_mix_w1.weight"), True)
        put(pth(p + "time_mix_w2.weight"), True)
        put(pth(p + "time_mix_a1.weight"), True)
        put(pth(p + "time_mix_a2.weight"), True)
        put(pth(p + "time_mix_v1.weight"), True)
        put(pth(p + "time_mix_v2.weight"), True)
        put(pth(p + "time_mix_g1.weight"), True)
        put(pth(p + "time_mix_g2.weight"), True)

        put(g.tensor(p + "time_mix_w0.weight"))
        put(g.tensor(p + "time_mix_a0.weight"))
        put(g.tensor(p + "time_mix_v0.weight"))
        put(g.tensor(p + "time_mix_k_k.weight"))
        put(g.tensor(p + "time_mix_k_a.weight"))
        put(g.tensor(p + "time_mix_r_k.weight"))
        put(g.tensor(p + "time_mix_ln.weight"))
        put(g.tensor(p + "time_mix_ln.bias"))

        put(t_of(pth(p + "channel_mix_key.weight")), True)
        put(t_of(pth(p + "channel_mix_value.weight")), True)
        put(g.tensor(p + "channel_mix_lerp_k.weight"))

    put(g.tensor("token_embd_norm.weight"))
    put(g.tensor("token_embd_norm.bias"))
    put(g.tensor("output_norm.weight"))
    put(g.tensor("output_norm.bias"))
    put(g.tensor("token_embd.weight"))          # 查表, 保持 fp32
    head = "output.weight" if g.has("output.weight") else "token_embd.weight"
    put(t_of(g.tensor(head)), True)             # LM head 走 INT4

    buf = b"".join(parts)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(buf)
    info = {"file": dst, "bytes": len(buf), "mb": round(len(buf) / 2**20, 1),
            "layers": L, "dim": D, "vocab": vocab, "int4": bool(use_int4),
            **stats}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())