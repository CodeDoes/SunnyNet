"""把 HF 的 DeltaNet 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 1 个 float32 + 3 个 int32, 其后为张量流):
  header: magic, version=1, layers, D, NH, HK, HV, F
  eps (float32)
  meta:   vocab, conv, tied_head
  张量流:
    per layer: attn_norm(D)
      q_proj k_proj v_proj o_proj  各 (out,in), 与 HF 原生布局一致
      q_conv1d k_conv1d (KV, conv), v_conv1d (VV, conv)
      o_norm(HV) b_proj(NH,D) mlp_norm(D)
      gate_proj up_proj (F,D)  down_proj (D,F)
    尾部: embed(vocab,D)  norm(D)  [head(vocab,D), tied 时不写]

矩阵保持 HF 的 (out,in) 行主序: C 侧 gemv 按输出通道并行, 每行连续读。

--int4 时 GEMV 矩阵走分组 INT4 (quant.c, block 64), 每个矩阵前置 int32 标签
(1=INT4, 0=fp32) 并换用 Q4 magic。短卷积核 (KV/VV, conv) 与 b_proj(NH,D)
不量化: 前者是卷积不是 GEMV, 后者 out=8 太小且体积可忽略。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = 0x4E4E4444
MAGIC_Q4 = 0x51444E44    # "QNDD"
VERSION = 1
BLOCK = 64

CONVS = ("attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_int4 = "--int4" in sys.argv
    src = argv[0] if argv else "models/deltanet-pure"
    if len(argv) > 1:
        dst = argv[1]
    else:
        dst = src + ("/deltanet.q4.bin" if use_int4 else "/deltanet.c.bin")
    src_p = Path(src)
    out_p = Path(dst)

    from sunnynet.runtime import quant as Q

    import torch
    from safetensors.torch import load_file

    f = src_p / "model.safetensors"
    sd = load_file(str(f)) if f.exists() else torch.load(
        src_p / "pytorch_model.bin", map_location="cpu", weights_only=True)
    w = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items()}

    D = w["model.embeddings.weight"].shape[1]
    vocab = w["model.embeddings.weight"].shape[0]
    L = len({k.split(".")[2] for k in w if k.startswith("model.layers.")})
    NH = w["model.layers.0.attn.b_proj.weight"].shape[0]
    HK = w["model.layers.0.attn.q_proj.weight"].shape[0] // NH
    HV = w["model.layers.0.attn.v_proj.weight"].shape[0] // NH
    F = w["model.layers.0.mlp.gate_proj.weight"].shape[0]
    conv = w["model.layers.0.attn.q_conv1d.weight"].shape[-1]
    tied = "lm_head.weight" not in w
    eps = 1e-6

    def tx(name, layer=None):
        k = f"model.layers.{layer}.{name}" if layer is not None else name
        a = w[k]
        if name in CONVS:
            return np.ascontiguousarray(a.reshape(a.shape[0], -1), dtype=np.float32)
        return np.ascontiguousarray(a.reshape(-1), dtype=np.float32)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    nmats = [0]
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8if3i", MAGIC if not use_int4 else MAGIC_Q4,
                             VERSION, L, D, NH, HK, HV, F,
                             float(eps), vocab, conv, int(tied)))

        def put_vec(arr):
            fh.write(arr.tobytes())

        def put_mat(arr, n_out, n_in):
            """GEMV 矩阵. fp32: 直接落盘; INT4: 前置标签, 量化后 (out,in)。"""
            if not use_int4:
                fh.write(arr.tobytes())
                return
            if n_in % BLOCK:
                fh.write(struct.pack("<i", 0))
                fh.write(arr.tobytes())
                return
            q, s = Q.quant_i4(arr.reshape(n_out, n_in), BLOCK)
            fh.write(struct.pack("<i", 1))
            fh.write(q.tobytes() + s.astype(np.float32).tobytes())
            nmats[0] += 1

        for l in range(L):
            # 顺序必须与 deltanet.c 的加载序列逐项对应
            put_vec(tx("attn_norm.weight", l))
            put_mat(tx("attn.q_proj.weight", l), D, D)
            put_mat(tx("attn.k_proj.weight", l), D, D)
            put_mat(tx("attn.v_proj.weight", l), D, D)
            put_mat(tx("attn.o_proj.weight", l), D, D)
            for c in CONVS:
                put_vec(tx(c, l))
            put_vec(tx("attn.o_norm.weight", l))
            put_vec(tx("attn.b_proj.weight", l))
            put_vec(tx("mlp_norm.weight", l))
            put_mat(tx("mlp.gate_proj.weight", l), F, D)
            put_mat(tx("mlp.up_proj.weight", l), F, D)
            put_mat(tx("mlp.down_proj.weight", l), D, F)
        put_vec(tx("model.embeddings.weight"))
        put_vec(tx("model.norm.weight"))
        if not tied:
            put_mat(tx("lm_head.weight"), vocab, D)

    n_param = (sum(a.size for a in w.values()))
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} F={F} conv={conv} "
          f"vocab={vocab} tied={tied} int4={use_int4}")
    print(f"参数 {n_param/1e6:.2f}M -> {out_p} "
          f"({out_p.stat().st_size/1e6:.1f} MB) int4_mats={nmats[0]}")


if __name__ == "__main__":
    main()