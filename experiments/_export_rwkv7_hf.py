"""FLA 命名的 safetensors 权重 -> C Runtime 打包 blob (v3).

不走 GGUF: llama.cpp 的 convert 需要额外下载, 而 FLA 已发布同权重.
张量名映射经 191M 两种格式逐张量比对确认 (max|diff| = 0), 见 _rwkv7_map_verify.py。

布局与 _export_rwkv7.py 完全一致:
  header: magic, version=3, layers, dim, head_size, n_head, ff, g_rank, vocab, q4
  eps (float), ranks: L * (Rw, Ra, Rv, Rg)
  张量流 (每项前置 1 int32 标志: 1=INT4, 0=fp32):
    per layer: ln1 ln1.b ln2 ln2.b lerp(6D)
      r,k,v,o 各存 M^T (D,D)
      w1 (D,Rw) w2 (Rw,D) a1 (D,Ra) a2 (Ra,D)
      v1 (D,Rv) v2 (Rv,D) g1 (D,Rg) g2 (Rg,D)
      w0 a0 v0 kk ka rk lnx_w lnx_b (各 D)
      cmk (D,F) cmv (F,D) cmk_lerp (D)
    尾部: en en.b on on.b (各 D), embed (vocab,D, fp32), head^T (D,vocab)

矩阵存法: 逻辑矩阵 M 满足 y = M @ x, 存 M^T (行主序 (len(x), len(y)))。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime import quant as Q  # noqa: E402

MAGIC = 0x4B57524B
VERSION = 3
BLOCK = 64


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    use_int4 = "--f32" not in sys.argv

    cfg = json.loads((src.parent / "config.json").read_text(encoding="utf-8"))
    L = int(cfg["num_hidden_layers"])
    D = int(cfg["hidden_size"])
    HS = int(cfg["head_dim"])
    NH = D // HS
    F = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    eps = float(cfg.get("norm_eps", 1e-5))
    Rw = int(cfg["decay_low_rank_dim"])
    Ra = int(cfg["a_low_rank_dim"])
    Rv = int(cfg["v_low_rank_dim"])
    Rg = int(cfg["gate_low_rank_dim"])

    sd = load_file(str(src), device="cpu")

    def t(name: str) -> np.ndarray:
        return sd[name].float().numpy()

    def tT(name: str) -> np.ndarray:
        return np.ascontiguousarray(sd[name].float().numpy().T)

    parts: list[bytes] = []
    parts.append(struct.pack("<10i", MAGIC, VERSION, L, D, HS, NH, F, Rg,
                             vocab, 1 if use_int4 else 0))
    parts.append(struct.pack("<f", eps))
    for i in range(L):
        has_v = f"model.layers.{i}.attn.v_lora.lora.0.weight" in sd
        parts.append(struct.pack("<4i", Rw, Ra, Rv if has_v else Rv, Rg))

    stats = {"int4_mats": 0, "f32_mats": 0}

    def put(a: np.ndarray, quantize: bool = False):
        a = np.ascontiguousarray(a, dtype=np.float32)
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
        p = f"model.layers.{i}.attn."
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.layers.{i}.attn_norm.weight").tobytes())
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.layers.{i}.attn_norm.bias").tobytes())
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.layers.{i}.ffn_norm.weight").tobytes())
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.layers.{i}.ffn_norm.bias").tobytes())
        # lerp 融合: 顺序 (r, w, k, v, a, g), 与 GGUF time_mix_lerp_fused 一致
        xr = t(p + "x_r").reshape(D)
        xw = t(p + "x_w").reshape(D)
        xk = t(p + "x_k").reshape(D)
        xv = t(p + "x_v").reshape(D)
        xa = t(p + "x_a").reshape(D)
        xg = t(p + "x_g").reshape(D)
        parts.append(struct.pack("<i", 0))
        parts.append(np.stack([xr, xw, xk, xv, xa, xg]).astype(np.float32).tobytes())

        put(tT(p + "r_proj.weight"), True)
        put(tT(p + "k_proj.weight"), True)
        put(tT(p + "v_proj.weight"), True)
        put(tT(p + "o_proj.weight"), True)

        # LoRA 也存 (in,out) = FLA 张量的转置
        put(tT(p + "w_lora.lora.0.weight"), True)
        put(tT(p + "w_lora.lora.2.weight"), True)
        put(tT(p + "a_lora.lora.0.weight"), True)
        put(tT(p + "a_lora.lora.2.weight"), True)
        if f"model.layers.{i}.attn.v_lora.lora.0.weight" in sd:
            put(tT(p + "v_lora.lora.0.weight"), True)
            put(tT(p + "v_lora.lora.2.weight"), True)
        else:
            # 第 0 层无 v_lora (v_first 直接取 v), 占位保持秩表一致
            put(np.zeros((D, Rv), dtype=np.float32), True)
            put(np.zeros((Rv, D), dtype=np.float32), True)
        put(tT(p + "g_lora.lora.0.weight"), True)
        put(tT(p + "g_lora.lora.2.weight"), True)

        parts.append(struct.pack("<i", 0))
        parts.append(t(p + "w_lora.lora.2.bias").tobytes())
        parts.append(struct.pack("<i", 0))
        parts.append(t(p + "a_lora.lora.2.bias").tobytes())
        parts.append(struct.pack("<i", 0))
        if f"model.layers.{i}.attn.v_lora.lora.2.bias" in sd:
            vb = t(p + "v_lora.lora.2.bias")
        else:
            vb = np.zeros(D, dtype=np.float32)
        parts.append(vb.astype(np.float32).tobytes())
        for nm in ("k_k", "k_a"):
            parts.append(struct.pack("<i", 0))
            parts.append(t(p + nm).reshape(D).astype(np.float32).tobytes())
        parts.append(struct.pack("<i", 0))
        parts.append(t(p + "r_k").reshape(D).astype(np.float32).tobytes())
        for nm in ("g_norm.weight", "g_norm.bias"):
            parts.append(struct.pack("<i", 0))
            parts.append(t(p + nm).reshape(D).astype(np.float32).tobytes())

        ff = f"model.layers.{i}.ffn."
        put(tT(ff + "key.weight"), True)
        put(tT(ff + "value.weight"), True)
        parts.append(struct.pack("<i", 0))
        parts.append(t(ff + "x_k").reshape(D).astype(np.float32).tobytes())

    for nm in ("pre_norm.weight", "pre_norm.bias"):
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.layers.0.{nm}").astype(np.float32).tobytes())
    for nm in ("norm.weight", "norm.bias"):
        parts.append(struct.pack("<i", 0))
        parts.append(t(f"model.{nm}").astype(np.float32).tobytes())
    parts.append(struct.pack("<i", 0))
    parts.append(sd["model.embeddings.weight"].float().numpy().astype(np.float32).tobytes())
    put(tT("lm_head.weight"), True)

    buf = b"".join(parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(buf)
    info = {"file": str(dst), "bytes": len(buf), "mb": round(len(buf) / 2**20, 1),
            "layers": L, "dim": D, "heads": NH, "ff": F, "vocab": vocab,
            "ranks": {"Rw": Rw, "Ra": Ra, "Rv": Rv, "Rg": Rg},
            "int4": bool(use_int4), **stats}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())