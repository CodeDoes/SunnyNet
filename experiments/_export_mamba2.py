"""把 HF 的 Mamba-2 权重导出为 C Runtime 的打包 blob.

布局 (前置 8 个 int32 + 1 个 float32 + 3 个 int32, 其后为纯 fp32 张量流):
  header: magic, version=1, layers, D, NH, HD, NS, NG
  eps (float32)
  meta:   vocab, conv, tied_head
  张量流:
    per layer: norm(D)
      in_proj(out,in)  (out = DI + CD + DT)
      conv1d  reshape 成 (CD, conv)
      conv1d.bias(CD)  A_log(NH)  D(NH)  dt_bias(NH)
      mixer.norm(DI)
      out_proj(out,in)  (out = D, in = DI)
    尾部: embed(vocab,D)  norm_f(D)  [lm_head(vocab,D), tied 时不写]

矩阵保持 HF 的 (out,in) 行主序: C 侧 gemv 按输出通道并行, 每行连续读。
conv1d.weight 在 HF 里是 (CD,1,conv), reshape 成 (CD,conv) 即 C 侧要的布局。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAGIC = 0x324E4D53
VERSION = 1


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/mamba2-130m"
    dst = sys.argv[2] if len(sys.argv) > 2 else src + "/mamba2.c.bin"
    src_p = Path(src)
    out_p = Path(dst)

    import torch

    f = src_p / "pytorch_model.bin"
    if f.exists():
        sd = torch.load(f, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        sd = load_file(str(src_p / "model.safetensors"))
    w = {k: v.detach().to(torch.float32).numpy() for k, v in sd.items()}
    # embedding 与 lm_head 可能共用同一块存储 (tie_embeddings); 转 numpy 前判
    tied = (sd["lm_head.weight"].data_ptr()
            == sd["backbone.embedding.weight"].data_ptr())

    D = w["backbone.embedding.weight"].shape[1]
    vocab = w["backbone.embedding.weight"].shape[0]
    L = len({k.split(".")[2] for k in w if k.startswith("backbone.layers.")})
    NH = w["backbone.layers.0.mixer.A_log"].shape[0]
    CD = w["backbone.layers.0.mixer.conv1d.weight"].shape[0]
    conv = w["backbone.layers.0.mixer.conv1d.weight"].shape[-1]
    DI = w["backbone.layers.0.mixer.out_proj.weight"].shape[1]
    HD = DI // NH
    # CD = DI + 2*NG*NS 只能定出乘积; 取 mamba_ssm 的 Mamba2 默认 NG=1
    NG = 1
    NS = (CD - DI) // (2 * NG)
    DT = NH
    assert DI + 2 * NG * NS == CD, (DI, NG, NS, CD)
    eps = 1e-5

    def tx(name, layer=None):
        k = f"backbone.layers.{layer}.{name}" if layer is not None else name
        a = w[k]
        if name.endswith("conv1d.weight"):
            return np.ascontiguousarray(a.reshape(a.shape[0], -1), dtype=np.float32)
        return np.ascontiguousarray(a.reshape(-1), dtype=np.float32)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as fh:
        fh.write(struct.pack("<8if3i", MAGIC, VERSION, L, D, NH, HD, NS, NG,
                             float(eps), vocab, conv, int(tied)))
        for l in range(L):
            # 顺序必须与 mamba2.c 的 take_f32 序列逐项对应
            fh.write(tx("norm.weight", l).tobytes())
            fh.write(tx("mixer.in_proj.weight", l).tobytes())
            fh.write(tx("mixer.conv1d.weight", l).tobytes())
            fh.write(tx("mixer.conv1d.bias", l).tobytes())
            fh.write(tx("mixer.A_log", l).tobytes())
            fh.write(tx("mixer.D", l).tobytes())
            fh.write(tx("mixer.dt_bias", l).tobytes())
            fh.write(tx("mixer.norm.weight", l).tobytes())
            fh.write(tx("mixer.out_proj.weight", l).tobytes())
        fh.write(tx("backbone.embedding.weight").tobytes())
        fh.write(tx("backbone.norm_f.weight").tobytes())
        if not tied:
            fh.write(tx("lm_head.weight").tobytes())

    n_param = sum(a.size for a in w.values())
    print(f"L={L} D={D} NH={NH} HD={HD} NS={NS} NG={NG} DI={DI} CD={CD} "
          f"conv={conv} vocab={vocab} tied={tied}")
    print(f"参数 {n_param/1e6:.2f}M -> {out_p} ({out_p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()