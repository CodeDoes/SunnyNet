"""在 x86 上执行真 NEON 源码 (experiments/_neon_shim/arm_neon.h 提供标量 shim).

本机没有 ARM 设备, nor qemu/docker/WSL; 但 scan_neon.c 的组内索引与门控语义
可以借 shim 在 x86 上跑一遍, 与参考实现对拍。只能排除"索引错但仍在界内"这类
缺陷, 不能证明真机上的浮点行为与性能。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SHIM = Path(__file__).resolve().parent
SRC = ROOT / "sunnynet" / "kernels" / "scan_neon.c"
OUT = SHIM / "neon_shim.dll"

FL = ctypes.POINTER(ctypes.c_float)
CI = ctypes.c_int


def compile_shim(cc: str) -> Path:
    cmd = [cc, "-O3", "-shared", "-fPIC", "-std=c11",
           "-I", str(SHIM), str(SRC), "-o", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"shim 编译失败:\n{r.stderr[:2000]}")
    return OUT


def ref_dw_with_scan(p, nu, s0, chunk):
    """参考: 逐元素衰减 s_new = p*s + (1-p)*nu, u_out 为输入 nu 本身."""
    B, T, N = p.shape
    out = np.empty_like(p)
    send = np.empty_like(s0)
    for b in range(B):
        s = s0[b].copy()
        for t in range(T):
            s = p[b, t] * s + (1.0 - p[b, t]) * nu[b, t]
            out[b, t] = s
        send[b] = s
    return out, send


def _pair_gate(e8):
    g1 = 1.0 / (1.0 + e8 / np.roll(e8, -1))
    g2 = 1.0 / (1.0 + e8 / np.roll(e8, -2))
    return g1 * g2


def ref_core_fwd(q, delta, s0):
    """参考: 组内成对竞争门控 + 门控概率 + 扫描 (对齐 layers.py::_compete)."""
    B, T, D2 = q.shape
    D = D2 // 2
    NG = D // 8
    u_raw = torch.from_numpy(q[..., :D].copy())
    gate = torch.from_numpy(q[..., D:].copy())
    dl = torch.from_numpy(delta.copy()).reshape(NG, 8)
    s = torch.from_numpy(s0.copy())
    g = gate.reshape(B, T, NG, 8)
    ur = u_raw.reshape(B, T, NG, 8)
    sig = torch.sigmoid(g)
    states = torch.empty(B, T, NG, 8)
    us = torch.empty(B, T, NG, 8)
    ps = torch.empty(B, T, NG, 8)
    w = torch.sigmoid(g - g.roll(-1, dims=-1)) * torch.sigmoid(g - g.roll(-2, dims=-1))
    for b in range(B):
        sb = s[b].reshape(NG, 8)
        for t in range(T):
            u = ur[b, t] * w[b, t]
            p = dl * sig[b, t] + (1.0 - dl)
            sb = p * sb + (1.0 - p) * u
            states[b, t], us[b, t], ps[b, t] = sb, u, p
    return (states.reshape(B, T, D).numpy(), us.reshape(B, T, D).numpy(),
            ps.reshape(B, T, D).numpy())


def ref_step_core(q, delta, s):
    """参考: 单步 (推理态) 的组内竞争门控 + 衰减, 对齐 layers.py::_compete."""
    B, D2 = q.shape
    D = D2 // 2
    NG = D // 8
    ur = q[:, :D].reshape(B, NG, 8)
    g = q[:, D:].reshape(B, NG, 8)
    sig = 1.0 / (1.0 + np.exp(-g))
    w = (1.0 / (1.0 + np.exp(-(g - np.roll(g, -1, axis=-1))))) * \
        (1.0 / (1.0 + np.exp(-(g - np.roll(g, -2, axis=-1)))))
    u = ur * w
    p = delta.reshape(NG, 8) * sig + (1.0 - delta.reshape(NG, 8))
    s_n = p * s.reshape(B, NG, 8) + (1.0 - p) * u
    return s_n.reshape(B, D), u.reshape(B, D)


def main() -> int:
    cc = sys.argv[1] if len(sys.argv) > 1 else "gcc"
    compile_shim(cc)
    lib = ctypes.CDLL(str(OUT))
    for fn, argt in (
        ("scan_neon_f32", [FL, FL, FL, FL, FL, CI, CI, CI]),
        ("core_neon_fwd_f32", [FL, FL, FL, FL, FL, FL, CI, CI, CI]),
        ("step_core_neon_f32", [FL, FL, FL, FL, FL, CI, CI]),
    ):
        if not hasattr(lib, fn):
            raise RuntimeError(f"shim 库缺少符号 {fn}")
        getattr(lib, fn).argtypes = argt

    rng = np.random.default_rng(2026)
    res: dict = {"compiled_from": str(SRC.relative_to(ROOT)),
                 "shim": "experiments/_neon_shim/arm_neon.h"}

    # --- scan ---
    B, T, N = 2, 6, 32
    p = rng.random((B, T, N), dtype=np.float32)
    nu = rng.standard_normal((B, T, N)).astype(np.float32)
    s0 = rng.standard_normal((B, N)).astype(np.float32)
    out = np.zeros_like(p)
    send = np.zeros_like(s0)
    lib.scan_neon_f32(p.ctypes.data_as(FL), nu.ctypes.data_as(FL),
                      s0.ctypes.data_as(FL), out.ctypes.data_as(FL),
                      send.ctypes.data_as(FL), B, T, N)
    r_out, r_send = ref_dw_with_scan(p, nu, s0, None)
    res["scan"] = {"maxdiff_states": float(np.abs(out - r_out).max()),
                   "maxdiff_send": float(np.abs(send - r_send).max())}

    # --- core fwd ---
    B, T, D = 2, 5, 16
    q = rng.standard_normal((B, T, 2 * D)).astype(np.float32)
    delta = rng.random((D,), dtype=np.float32).astype(np.float32)
    s0c = rng.standard_normal((B, D)).astype(np.float32)
    st = np.zeros((B, T, D), dtype=np.float32)
    uo = np.zeros((B, T, D), dtype=np.float32)
    po = np.zeros((B, T, D), dtype=np.float32)
    lib.core_neon_fwd_f32(q.ctypes.data_as(FL), delta.ctypes.data_as(FL),
                          s0c.ctypes.data_as(FL), st.ctypes.data_as(FL),
                          uo.ctypes.data_as(FL), po.ctypes.data_as(FL),
                          B, T, D)
    r_st, r_uo, r_po = ref_core_fwd(q, delta, s0c)
    res["core_fwd"] = {"maxdiff_states": float(np.abs(st - r_st).max()),
                       "maxdiff_u": float(np.abs(uo - r_uo).max()),
                       "maxdiff_p": float(np.abs(po - r_po).max())}

    # --- step core ---
    B, D = 2, 16
    q = rng.standard_normal((B, 2 * D)).astype(np.float32)
    delta = rng.random((D,), dtype=np.float32)
    sc = rng.standard_normal((B, D)).astype(np.float32)
    sn = np.zeros((B, D), dtype=np.float32)
    us = np.zeros((B, D), dtype=np.float32)
    lib.step_core_neon_f32(q.ctypes.data_as(FL), delta.ctypes.data_as(FL),
                           sc.ctypes.data_as(FL), sn.ctypes.data_as(FL),
                           us.ctypes.data_as(FL), B, D)
    r_sn, r_us = ref_step_core(q, delta, sc)
    res["step_core"] = {"maxdiff_states": float(np.abs(sn - r_sn).max()),
                        "maxdiff_u": float(np.abs(us - r_us).max())}

    print(json.dumps(res, ensure_ascii=False, indent=2))
    ok = all(v < 1e-5 for d in res.values() if isinstance(d, dict)
             for k, v in d.items())
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())