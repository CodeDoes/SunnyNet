"""把 HF 的 HGRN2 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 4 个 float32 + 5 个 int32, 其后为纯 fp32 张量流):
  header: magic, version=1, layers, D, NH, HK, HV, unused
  floats: eps, scale, 0, lb_flag
  meta:   vocab, F, tied_head, 0, 0
  张量流:
    per layer: lb(D)  attn_norm(D)
      q_proj f_proj i_proj (D,D)   o_proj (D,D)
      g_norm(D)  mlp_norm(D)
      gate_proj up_proj (F,D)  down_proj (D,F)
    尾部: embed(vocab,D)  norm(D)  [head(vocab,D), tied 时不写]

与 GLA 的差异: 无 k_proj (k = 1 - exp(g) 由 f_proj 导出), 无低秩 gk_proj,
无输出门 (改整体 RMSNorm), 逐层学习下界 lb。
矩阵保持 HF 的 (out,in) 行主序; 权重源是 bf16, 统一转 fp32 落盘。

--int4 时 GEMV 矩阵走分组 INT4 (quant.c, block 64), 每个矩阵前置 int32 标签
(1=INT4, 0=fp32) 并换用 Q4 magic; 向量与嵌入表不量化 (嵌入是查表不是乘加)。
敏感矩阵可用 --keep-f32=fproj,head,qproj,oproj,mlp 单独保 fp32:
  fproj 决定 decay g=logsigmoid(f), 误差经 exp 放大后再累积进状态;
  head  直接决定 argmax。两种 blob 的名字与 magic 都不同, 互不影响。

权重 2.7GB (bf16), 转 fp32 是 5.4GB, 本机空闲内存不足以整份驻留,
故逐张量读转写, 峰值只占单张 (最大 32000x2048 = 262MB)。
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = 0x32474853       # "SHG2" fp32
MAGIC_Q4 = 0x51474853    # "SHGQ" INT4 带标签
VERSION = 1
BLOCK = 64


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_int4 = "--int4" in sys.argv
    keep = set()
    for a in sys.argv[1:]:
        if a.startswith("--keep-f32"):
            keep |= {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    src = argv[0] if argv else "models/hgrn2-1.3B-100B"
    if len(argv) > 1:
        dst = argv[1]
    else:
        tag = ("-" + "-".join(sorted(keep))) if keep else ""
        dst = src + ("/hgrn2.q4%s.bin" % tag if use_int4 else "/hgrn2.c.bin")
    src_p = Path(src)
    out_p = Path(dst)

    from sunnynet.runtime import quant as Q

    import torch
    from safetensors import safe_open

    f = src_p / "model.safetensors"
    sd = None
    if f.exists():
        h = safe_open(str(f), framework="pt")
        keys = set(h.keys())
    else:
        sd = torch.load(src_p / "pytorch_model.bin", map_location="cpu",
                        weights_only=True)
        h = None
        keys = set(sd)

    def raw(key):
        return h.get_tensor(key) if h is not None else sd[key]

    cfg = json.loads((src_p / "config.json").read_text(encoding="utf-8"))
    emb = raw("model.embeddings.weight")
    D = emb.shape[1]
    vocab = emb.shape[0]
    del emb

    L = int(cfg["num_hidden_layers"])
    GR = int(cfg["expand_ratio"])
    NH = D // GR
    KD = raw("model.layers.0.attn.q_proj.weight").shape[0]
    VD = raw("model.layers.0.attn.i_proj.weight").shape[0]
    HK = KD // NH
    HV = VD // NH
    F = raw("model.layers.0.mlp.gate_proj.weight").shape[0]
    eps = float(cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-6)))
    scale = HK ** -0.5
    use_lb = bool(cfg.get("use_lower_bound", False))

    # lm_head 是否与 embeddings 同值: 比首尾若干行 (data_ptr 在逐张量读下不可得)
    if "lm_head.weight" not in keys:
        tied = True
    else:
        a = raw("model.embeddings.weight").to(torch.float32)
        b = raw("lm_head.weight").to(torch.float32)
        tied = bool(a.shape == b.shape
                    and torch.equal(a[:4], b[:4])
                    and torch.equal(a[-4:], b[-4:]))
        del a, b

    # 层循环里 lb = softmax(lower_bounds, dim=0).cumsum(0) - softmax(...)[0]
    if use_lb:
        lb = raw("model.lower_bounds").to(torch.float32).numpy()
        sm = np.exp(lb - lb.max(axis=0, keepdims=True))
        sm = sm / sm.sum(axis=0, keepdims=True)
        lb_all = (np.cumsum(sm, axis=0) - sm[0]).astype(np.float32)
        del lb, sm
    else:
        lb_all = None

    def tx(name, layer=None):
        k = f"model.layers.{layer}.{name}" if layer is not None else name
        return np.ascontiguousarray(
            raw(k).to(torch.float32).numpy().reshape(-1), dtype=np.float32)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    nmats = [0]
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8i4f5i", MAGIC if not use_int4 else MAGIC_Q4,
                             VERSION, L, D, NH, HK, HV, 0,
                             float(eps), float(scale), 0.0, float(use_lb),
                             vocab, F, int(tied), 0, 0))

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
            # 顺序必须与 hgrn2.c 的加载序列逐项对应
            put_vec(lb_all[l] if lb_all is not None
                    else np.zeros(D, np.float32))
            put_vec(tx("attn_norm.weight", l))
            put_mat(tx("attn.q_proj.weight", l), KD, D, "qproj")
            put_mat(tx("attn.f_proj.weight", l), KD, D, "fproj")
            put_mat(tx("attn.i_proj.weight", l), VD, D, "iproj")
            put_mat(tx("attn.o_proj.weight", l), D, VD, "oproj")
            put_vec(tx("attn.g_norm.weight", l))
            put_vec(tx("mlp_norm.weight", l))
            put_mat(tx("mlp.gate_proj.weight", l), F, D, "mlp")
            put_mat(tx("mlp.up_proj.weight", l), F, D, "mlp")
            put_mat(tx("mlp.down_proj.weight", l), D, F, "mlp")
        put_vec(tx("model.embeddings.weight"))
        put_vec(tx("model.norm.weight"))
        if not tied:
            put_mat(tx("lm_head.weight"), vocab, D, "head")

    n_param = 4 * L * D * D + 3 * L * F * D + 2 * vocab * D + 4 * L * D
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} KD={KD} VD={VD} F={F} "
          f"vocab={vocab} tied={tied} use_lb={use_lb} int4={use_int4} "
          f"keep={sorted(keep)}")
    print(f"参数约 {n_param/1e6:.1f}M -> {out_p} "
          f"({out_p.stat().st_size/1e6:.1f} MB) int4_mats={nmats[0]}")


if __name__ == "__main__":
    main()