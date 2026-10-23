"""Mamba-2 逐层对表: 定位误差落在哪一层。

复用 numpy 参考的前向逻辑, 每层后记录主流 x, 与 C 侧 sn_m2_dbg 对照。
用法: python -X utf8 experiments/_mamba2_layer_probe.py
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))
BLOB = Path(MDIR) / "mamba2.c.bin"


def ref_layerwise(m, token):
    """走一遍 numpy 参考, 返回每层后的 x (copy)。"""
    from sunnynet.runtime.mamba2 import _silu, _softplus, _rms_norm

    NH, HD, NS, NG = m.n_heads, m.head_dim, m.state_size, m.n_groups
    out = []
    x = m.w("backbone.embedding.weight")[token].astype(np.float32)
    conv_all = np.zeros((m.n_layers, m.conv_kernel - 1, m.conv_dim), dtype=np.float32)
    S_all = np.zeros((m.n_layers, NH, HD, NS), dtype=np.float32)

    for i in range(m.n_layers):
        p = f"backbone.layers.{i}."
        res = x
        h = _rms_norm(x, m.w(p + "norm.weight"), m.eps)
        proj = m.w(p + "mixer.in_proj.weight") @ h
        gate = proj[: m.d_inner]
        bc = proj[m.d_inner: m.d_inner + m.conv_dim]
        dt = proj[m.d_inner + m.conv_dim:]

        hist = conv_all[i]
        kw = m.w(p + "mixer.conv1d.weight").reshape(m.conv_dim, m.conv_kernel)
        kb = m.w(p + "mixer.conv1d.bias")
        acc = np.zeros(m.conv_dim, dtype=np.float32)
        for t in range(m.conv_kernel - 1):
            acc += kw[:, t] * hist[t]
        acc += kw[:, -1] * bc
        if m.conv_kernel > 1:
            hist[:-1] = hist[1:]
            hist[-1] = bc
        act = _silu(acc + kb)

        x_ssm = act[: m.d_inner].reshape(NH, HD)
        B = act[m.d_inner: m.d_inner + NG * NS].reshape(NG, NS)
        C = act[m.d_inner + NG * NS:].reshape(NG, NS)

        A = -np.exp(m.w(p + "mixer.A_log"))
        dth = _softplus(dt + m.w(p + "mixer.dt_bias"))
        dA = np.exp(dth[:, None] * A[:, None])
        Bg = np.repeat(B, NH // NG, axis=0)
        Cg = np.repeat(C, NH // NG, axis=0)
        dBx = (dth[:, None] * Bg)[:, None, :] * x_ssm[:, :, None]

        S = S_all[i] * dA[:, :, None] + dBx
        S_all[i] = S
        y = np.einsum("hpn,hn->hp", S, Cg)
        y = y + m.w(p + "mixer.D")[:, None] * x_ssm
        y = y.reshape(-1) * _silu(gate)
        y = _rms_norm(y, m.w(p + "mixer.norm.weight"), m.eps)
        x = res + m.w(p + "mixer.out_proj.weight") @ y
        out.append(x.copy())
    return out


def main():
    from sunnynet.runtime.mamba2 import Mamba2
    from sunnynet.runtime import mamba2_c
    from sunnynet.runtime.mamba2_c import SnMamba2

    m = Mamba2(MDIR)
    h = SnMamba2(BLOB, m.vocab)
    D = m.d_model
    lib = mamba2_c._load()
    lib.sn_m2_dbg.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_float)]
    buf = (ctypes.c_float * D)()

    for tok in (1, 415, 5565):
        refs = ref_layerwise(m, tok)
        print(f"--- token={tok} ---")
        for l in range(m.n_layers):
            h.reset()          # dbg 会推进状态, 每次都要从零起
            lib.sn_m2_dbg(h._h, int(tok), l, buf)
            c = np.frombuffer(bytes(buf), dtype=np.float32)
            d = float(np.abs(c - refs[l]).max())
            scale = float(np.abs(refs[l]).max())
            print(f"  layer {l:2d}: max|diff|={d:.3e}  |x|max={scale:.3f} "
                  f"rel={d/max(scale,1e-9):.2e}")
        h.reset()


if __name__ == "__main__":
    main()