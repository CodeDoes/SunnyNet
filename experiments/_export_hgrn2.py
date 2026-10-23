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

MAGIC = 0x32474853   # "SHG2"
VERSION = 1


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/hgrn2-1.3B-100B"
    dst = sys.argv[2] if len(sys.argv) > 2 else src + "/hgrn2.c.bin"
    src_p = Path(src)
    out_p = Path(dst)

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
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8i4f5i", MAGIC, VERSION, L, D, NH, HK, HV, 0,
                             float(eps), float(scale), 0.0, float(use_lb),
                             vocab, F, int(tied), 0, 0))
        for l in range(L):
            # 顺序必须与 hgrn2.c 的 take_f32 序列逐项对应
            fh.write((lb_all[l] if lb_all is not None
                      else np.zeros(D, np.float32)).tobytes())
            fh.write(tx("attn_norm.weight", l).tobytes())
            for m in ("attn.q_proj.weight", "attn.f_proj.weight",
                      "attn.i_proj.weight", "attn.o_proj.weight"):
                fh.write(tx(m, l).tobytes())
            for m in ("attn.g_norm.weight", "mlp_norm.weight"):
                fh.write(tx(m, l).tobytes())
            for m in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                      "mlp.down_proj.weight"):
                fh.write(tx(m, l).tobytes())
        fh.write(tx("model.embeddings.weight").tobytes())
        fh.write(tx("model.norm.weight").tobytes())
        if not tied:
            fh.write(tx("lm_head.weight").tobytes())

    n_param = 4 * L * D * D + 3 * L * F * D + 2 * vocab * D + 4 * L * D
    print(f"L={L} D={D} NH={NH} HK={HK} HV={HV} KD={KD} VD={VD} F={F} "
          f"vocab={vocab} tied={tied} use_lb={use_lb}")
    print(f"参数约 {n_param/1e6:.1f}M -> {out_p} "
          f"({out_p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()