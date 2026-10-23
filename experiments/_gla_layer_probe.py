"""GLA 逐层对表: 定位误差落在哪一层.

用法: python -X utf8 experiments/_gla_layer_probe.py
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


def ref_layerwise(m, token):
    """走一遍 numpy 参考, 返回每层后的 x (copy)。"""
    from sunnynet.runtime.gla import _silu, _logsigmoid, _rms_norm

    H, K, V = m.n_heads, m.head_k_dim, m.head_v_dim
    out = []
    x = m.w("model.embeddings.weight")[token].astype(np.float32)
    S_all = np.zeros((m.n_layers, H, K, V), dtype=np.float32)

    for i in range(m.n_layers):
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
        S = S_all[i] * decay[:, :, None] + k2[:, :, None] * v2[:, None, :]
        S_all[i] = S
        o = np.einsum("hkv,hk->hv", S, q2) * m.scale
        g = m.w(p + "attn.g_proj.weight") @ h
        o = _rms_norm(o, m.w(p + "attn.g_norm_swish_gate.weight"), m.eps)
        o = (o * _silu(g.reshape(H, V))).reshape(-1)
        x = res + m.w(p + "attn.o_proj.weight") @ o
        h2 = _rms_norm(x, m.w(p + "mlp_norm.weight"), m.eps)
        x = x + m.w(p + "mlp.down_proj.weight") @ (
            _silu(m.w(p + "mlp.gate_proj.weight") @ h2)
            * (m.w(p + "mlp.up_proj.weight") @ h2))
        out.append(x.copy())
    return out


def main():
    from sunnynet.runtime.gla import GLA
    from sunnynet.runtime import gla_c
    from sunnynet.runtime.gla_c import SnGLA

    m = GLA(MDIR)
    h = SnGLA(BLOB, m.vocab)
    D = m.hidden
    lib = gla_c._load()
    lib.sn_gla_dbg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float)]
    buf = (ctypes.c_float * D)()

    for tok in (1, 415):
        refs = ref_layerwise(m, tok)
        print(f"--- token={tok} ---")
        for l in range(m.n_layers):
            h.reset()          # dbg 会推进状态, 每次都要从零起
            lib.sn_gla_dbg(h._h, int(tok), l, buf)
            c = np.frombuffer(bytes(buf), dtype=np.float32)
            d = float(np.abs(c - refs[l]).max())
            scale = float(np.abs(refs[l]).max())
            print(f"  layer {l:2d}: max|diff|={d:.3e}  |x|max={scale:.3f} "
                  f"rel={d/max(scale,1e-9):.2e}")
        h.reset()


if __name__ == "__main__":
    main()