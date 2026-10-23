"""把 HF 的 GLA 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 4 个 float32 + 5 个 int32, 其后为纯 fp32 张量流):
  header: magic, version=1, layers, D, NH, HK, HV, GR
  floats: eps, scale, gate_logit_normalizer, clamp_min
  meta:   vocab, F, tied_head, clamp_min_set, 0
  张量流:
    per layer: attn_norm(D)
      q_proj k_proj (KD,D)   v_proj o_proj (VD,D)
      gk_proj.0 (GR,D)  gk_proj.1.weight (KD,GR)  gk_proj.1.bias (KD,)
      g_proj (VD,D)  g_norm_swish_gate (HV,)
      mlp_norm(D)
      gate_proj up_proj (F,D)  down_proj (D,F)
    尾部: embed(vocab,D)  norm(D)  [head(vocab,D), tied 时不写]

矩阵保持 HF 的 (out,in) 行主序: C 侧 gemv 按输出通道并行, 每行连续读。
权重源是 bf16, 统一转 fp32 落盘。
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = 0x414C4753
VERSION = 1


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/gla-340M"
    dst = sys.argv[2] if len(sys.argv) > 2 else src + "/gla.c.bin"
    src_p = Path(src)
    out_p = Path(dst)

    import torch
    from safetensors.torch import load_file

    f = src_p / "model.safetensors"
    if f.exists():
        sd = load_file(str(f))
    else:
        sd = torch.load(src_p / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
    # embedding 与 lm_head 可能共用同一块存储; 转 numpy 前判共享
    tied = ("lm_head.weight" not in sd
            or sd["lm_head.weight"].data_ptr()
            == sd["model.embeddings.weight"].data_ptr())
    w = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items()}

    cfg = json.loads((src_p / "config.json").read_text(encoding="utf-8"))
    D = w["model.embeddings.weight"].shape[1]
    vocab = w["model.embeddings.weight"].shape[0]
    L = len({k.split(".")[2] for k in w if k.startswith("model.layers.")})
    NH = int(cfg["num_heads"])
    KD = w["model.layers.0.attn.q_proj.weight"].shape[0]
    VD = w["model.layers.0.attn.v_proj.weight"].shape[0]
    HK = KD // NH
    HV = VD // NH
    GR = w["model.layers.0.attn.gk_proj.0.weight"].shape[0]
    F = w["model.layers.0.mlp.gate_proj.weight"].shape[0]
    eps = float(cfg.get("norm_eps", 1e-6))
    gnorm = float(cfg.get("gate_logit_normalizer", 16))
    scale = HK ** -0.5
    cm = cfg.get("clamp_min", None)
    clamp_set = 1 if cm is not None else 0
    clamp_min = float(cm) if cm is not None else 0.0

    def tx(name, layer=None):
        k = f"model.layers.{layer}.{name}" if layer is not None else name
        return np.ascontiguousarray(w[k].reshape(-1), dtype=np.float32)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8i4f5i", MAGIC, VERSION, L, D, NH, HK, HV, GR,
                             float(eps), float(scale), float(gnorm),
                             float(clamp_min), vocab, F, int(tied),
                             clamp_set, 0))
        for l in range(L):
            # 顺序必须与 gla.c 的 take_f32 序列逐项对应
            fh.write(tx("attn_norm.weight", l).tobytes())
            for m in ("attn.q_proj.weight", "attn.k_proj.weight",
                      "attn.v_proj.weight", "attn.o_proj.weight"):
                fh.write(tx(m, l).tobytes())
            for m in ("attn.gk_proj.0.weight", "attn.gk_proj.1.weight",
                      "attn.gk_proj.1.bias"):
                fh.write(tx(m, l).tobytes())
            for m in ("attn.g_proj.weight", "attn.g_norm_swish_gate.weight",
                      "mlp_norm.weight"):
                fh.write(tx(m, l).tobytes())
            for m in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"):
                fh.write(tx(m, l).tobytes())
        fh.write(tx("model.embeddings.weight").tobytes())
        fh.write(tx("model.norm.weight").tobytes())
        if not tied:
            fh.write(tx("lm_head.weight").tobytes())

    n_param = sum(a.size for a in w.values())
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} GR={GR} KD={KD} VD={VD} F={F} "
          f"vocab={vocab} tied={tied} clamp_min={cm}")
    print(f"参数 {n_param/1e6:.2f}M -> {out_p} ({out_p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()