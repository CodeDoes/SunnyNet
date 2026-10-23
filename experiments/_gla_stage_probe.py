"""GLA 层内阶段对表: 定位第 layer 层里哪个张量偏了.

用法: python -X utf8 experiments/_gla_stage_probe.py [layer]
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_GLA", str(ROOT / "models" / "gla-340M"))
BLOB = Path(MDIR) / "gla.c.bin"

STAGES = ["attn_norm", "q_proj", "k_proj", "v_proj", "exp(gk)", "g_proj",
          "readout", "RMSNorm", "gated", "o_proj"]


def ref_stages(m, token, layer):
    """numpy 参考: 单步跑到指定层的各阶段中间量。"""
    from sunnynet.runtime.gla import _silu, _logsigmoid, _rms_norm

    H, K, V = m.n_heads, m.head_k_dim, m.head_v_dim
    x = m.w("model.embeddings.weight")[token].astype(np.float32)
    for i in range(layer + 1):
        p = f"model.layers.{i}."
        res = x
        h = _rms_norm(x, m.w(p + "attn_norm.weight"), m.eps)
        q = m.w(p + "attn.q_proj.weight") @ h
        k = m.w(p + "attn.k_proj.weight") @ h
        v = m.w(p + "attn.v_proj.weight") @ h
        gk = (m.w(p + "attn.gk_proj.1.weight")
              @ (m.w(p + "attn.gk_proj.0.weight") @ h)
              + m.w(p + "attn.gk_proj.1.bias"))
        gk = _logsigmoid(gk) / m.gate_logit_normalizer
        if m.clamp_min is not None:
            gk = np.maximum(gk, m.clamp_min)
        q2, k2, v2 = q.reshape(H, K), k.reshape(H, K), v.reshape(H, V)
        decay = np.exp(gk.reshape(H, K), dtype=np.float32)
        # 单步: S 从零起, 故 o = (k⊗v)^T q * scale = k 与 q 的内积 * v
        S = k2[:, :, None] * v2[:, None, :]
        o = np.einsum("hkv,hk->hv", S, q2) * m.scale
        g = m.w(p + "attn.g_proj.weight") @ h
        o_n = _rms_norm(o, m.w(p + "attn.g_norm_swish_gate.weight"), m.eps)
        o_g = (o_n * _silu(g.reshape(H, V))).reshape(-1).copy()
        op = m.w(p + "attn.o_proj.weight") @ o_g
        if i == layer:
            return {"attn_norm": h, "q_proj": q, "k_proj": k, "v_proj": v,
                    "exp(gk)": decay.reshape(-1), "g_proj": g,
                    "readout": o.reshape(-1), "RMSNorm": o_n.reshape(-1),
                    "gated": o_g, "o_proj": op}
        x = res + op
        h2 = _rms_norm(x, m.w(p + "mlp_norm.weight"), m.eps)
        x = x + m.w(p + "mlp.down_proj.weight") @ (
            _silu(m.w(p + "mlp.gate_proj.weight") @ h2)
            * (m.w(p + "mlp.up_proj.weight") @ h2))
    return None


def main():
    from sunnynet.runtime.gla import GLA
    from sunnynet.runtime import gla_c
    from sunnynet.runtime.gla_c import SnGLA

    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    m = GLA(MDIR)
    h = SnGLA(BLOB, m.vocab)
    lib = gla_c._load()
    lib.sn_gla_probe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    D = m.hidden
    buf = (ctypes.c_float * (m.value_dim + 8))()

    for tok in (1, 415):
        refs = ref_stages(m, tok, layer)
        print(f"--- token={tok} layer={layer} ---")
        for si, name in enumerate(STAGES):
            h.reset()
            lib.sn_gla_probe(h._h, int(tok), layer, si, buf)
            c = np.frombuffer(bytes(buf), dtype=np.float32)[: refs[name].size]
            d = float(np.abs(c - refs[name]).max())
            sc = float(np.abs(refs[name]).max())
            flag = "  <== 偏差" if d / max(sc, 1e-9) > 1e-4 else ""
            print(f"  {name:10s}: max|diff|={d:.3e}  |ref|max={sc:.3f} "
                  f"rel={d/max(sc,1e-9):.2e}{flag}")
        h.reset()


if __name__ == "__main__":
    main()