"""把 HF 的 DeltaNet 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 1 个 float32 + 3 个 int32, 其后为纯 fp32 张量流):
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
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = 0x4E4E4444
VERSION = 1

CONVS = ("attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/deltanet-pure"
    dst = sys.argv[2] if len(sys.argv) > 2 else src + "/deltanet.c.bin"
    src_p = Path(src)
    out_p = Path(dst)

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
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8if3i", MAGIC, VERSION, L, D, NH, HK, HV, F,
                             float(eps), vocab, conv, int(tied)))
        for l in range(L):
            # 顺序必须与 deltanet.c 的 take_f32 序列逐项对应
            fh.write(tx("attn_norm.weight", l).tobytes())
            for m in ("attn.q_proj.weight", "attn.k_proj.weight",
                      "attn.v_proj.weight", "attn.o_proj.weight"):
                fh.write(tx(m, l).tobytes())
            for c in CONVS:
                fh.write(tx(c, l).tobytes())
            fh.write(tx("attn.o_norm.weight", l).tobytes())
            fh.write(tx("attn.b_proj.weight", l).tobytes())
            fh.write(tx("mlp_norm.weight", l).tobytes())
            for m in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"):
                fh.write(tx(m, l).tobytes())
        fh.write(tx("model.embeddings.weight").tobytes())
        fh.write(tx("model.norm.weight").tobytes())
        if not tied:
            fh.write(tx("lm_head.weight").tobytes())

    n_param = (sum(a.size for a in w.values()))
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} F={F} conv={conv} "
          f"vocab={vocab} tied={tied}")
    print(f"参数 {n_param/1e6:.2f}M -> {out_p} ({out_p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()