"""把 HF 的 GLA 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 4 个 float32 + 5 个 int32, 其后为张量流):
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

--int4 时 GEMV 矩阵走分组 INT4 (quant.c, block 64), 每个矩阵前置 int32 标签
(1=INT4, 0=fp32) 并换用 Q4 magic; 向量不量化。gk_proj.1.weight 的 in=GR=16
不满足 block, 自动回落 fp32。敏感矩阵可用 --keep-f32 单列 (类名同 BLOCK 注释)。
fp32 布局保持原样, 两种 blob 的文件名与 magic 都不同, 互不影响。
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
MAGIC_Q4 = 0x51414C47     # "QLAG"
VERSION = 1
BLOCK = 64


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_int4 = "--int4" in sys.argv
    keep = set()
    for a in sys.argv[1:]:
        if a.startswith("--keep-f32"):
            keep |= {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    src = argv[0] if argv else "models/gla-340M"
    if len(argv) > 1:
        dst = argv[1]
    else:
        tag = ("-" + "-".join(sorted(keep))) if keep else ""
        dst = src + ("/gla.q4%s.bin" % tag if use_int4 else "/gla.c.bin")
    src_p = Path(src)
    out_p = Path(dst)

    from sunnynet.runtime import quant as Q

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
    nmats = [0]
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8i4f5i", MAGIC if not use_int4 else MAGIC_Q4,
                             VERSION, L, D, NH, HK, HV, GR,
                             float(eps), float(scale), float(gnorm),
                             float(clamp_min), vocab, F, int(tied),
                             clamp_set, 0))

        def put_vec(arr):
            fh.write(arr.tobytes())

        def put_mat(arr, n_out, n_in, kind):
            """GEMV 矩阵. fp32: 直接落盘; INT4: 前置标签, 量化后 (out,in)。"""
            if not use_int4 or kind in keep:
                if use_int4:
                    fh.write(struct.pack("<i", 0))
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
            # 顺序必须与 gla.c 的加载序列逐项对应
            put_vec(tx("attn_norm.weight", l))
            put_mat(tx("attn.q_proj.weight", l), KD, D, "qproj")
            put_mat(tx("attn.k_proj.weight", l), KD, D, "kproj")
            put_mat(tx("attn.v_proj.weight", l), VD, D, "vproj")
            put_mat(tx("attn.o_proj.weight", l), D, VD, "oproj")
            put_mat(tx("attn.gk_proj.0.weight", l), GR, D, "gk")
            put_mat(tx("attn.gk_proj.1.weight", l), KD, GR, "gk")
            put_vec(tx("attn.gk_proj.1.bias", l))
            put_mat(tx("attn.g_proj.weight", l), VD, D, "gproj")
            put_vec(tx("attn.g_norm_swish_gate.weight", l))
            put_vec(tx("mlp_norm.weight", l))
            put_mat(tx("mlp.gate_proj.weight", l), F, D, "mlp")
            put_mat(tx("mlp.up_proj.weight", l), F, D, "mlp")
            put_mat(tx("mlp.down_proj.weight", l), D, F, "mlp")
        put_vec(tx("model.embeddings.weight"))
        put_vec(tx("model.norm.weight"))
        if not tied:
            put_mat(tx("lm_head.weight"), vocab, D, "head")

    n_param = sum(a.size for a in w.values())
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} GR={GR} KD={KD} VD={VD} F={F} "
          f"vocab={vocab} tied={tied} clamp_min={cm} int4={use_int4} "
          f"keep={sorted(keep)}")
    print(f"参数 {n_param/1e6:.2f}M -> {out_p} "
          f"({out_p.stat().st_size/1e6:.1f} MB) int4_mats={nmats[0]}")


if __name__ == "__main__":
    main()