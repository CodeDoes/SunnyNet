"""AVX2 与 AVX-512 内核的数值等价验证 + 性能对比.

本机无 AVX-512 时只能走到编译检查; 在支持 AVX-512 的机器上把三条路径
(torch 参考 / AVX2 / AVX-512) 对齐并分别计时。

用法:
    python -X utf8 experiments/_bench_avx512.py            # 编译检查 + 能跑则跑
    python -X utf8 experiments/_bench_avx512.py --check    # 只做编译检查
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FL = ctypes.POINTER(ctypes.c_float)
CI = ctypes.c_int


def _p(a: np.ndarray):
    return a.ctypes.data_as(FL)


def ref_core_fwd(q, delta, s0):
    """torch 参考: 组内成对竞争门控 + 门控概率 + 扫描."""
    import torch
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
    for b in range(B):
        sb = s[b].reshape(NG, 8)
        for t in range(T):
            gt = g[b, t]
            e = torch.exp(-gt)
            w = (1.0 / (1.0 + e / e.roll(1, dims=-1))) * \
                (1.0 / (1.0 + e / e.roll(2, dims=-1)))
            u = ur[b, t] * w
            p = dl * sig[b, t] + (1.0 - dl)
            sb = p * sb + (1.0 - p) * u
            states[b, t], us[b, t], ps[b, t] = sb, u, p
    return (states.reshape(B, T, D).numpy(), us.reshape(B, T, D).numpy(),
            ps.reshape(B, T, D).numpy())


def _avx2_lib():
    from sunnynet.kernels import BUILD
    name = "sunnynet_kernels.dll" if os.name == "nt" else "sunnynet_kernels.so"
    dll = BUILD / name
    if not dll.exists():
        return None, f"未找到 {dll}"
    lib = ctypes.CDLL(str(dll))
    lib.core_fwd_f32.argtypes = [FL, FL, FL, FL, FL, FL, CI, CI, CI]
    return lib, None


def _avx512_lib():
    from sunnynet.kernels import simd
    return simd.load_avx512_core(), None


def run(lib, q, delta, s0, B, T, D, rounds):
    st = np.empty((B, T, D), np.float32)
    uo = np.empty((B, T, D), np.float32)
    po = np.empty((B, T, D), np.float32)
    args = (_p(q), _p(delta), _p(s0), _p(st), _p(uo), _p(po), B, T, D)
    lib.core_fwd_f32(*args)
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        lib.core_fwd_f32(*args)
        ts.append((time.perf_counter() - t0) * 1e3)
    return st, uo, po, float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做编译检查")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    from sunnynet.kernels import simd
    info = {"arch": simd.arch_name(),
            "avx512_supported": simd.avx512_supported()}
    ok, msg = simd.avx512_core_compiles()
    info["core512_compiles"] = ok
    info["core512_build"] = msg
    if not ok or args.check:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0 if ok else 1

    B, T, D = args.B, args.T, args.D
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, T, 2 * D)) * 0.5).astype(np.float32)
    delta = rng.random(D).astype(np.float32)
    s0 = rng.standard_normal((B, D)).astype(np.float32)

    st_ref, u_ref, p_ref = ref_core_fwd(q, delta, s0)
    res = {}
    for name, loader in (("avx2", _avx2_lib), ("avx512", _avx512_lib)):
        try:
            lib, err = loader()
            if lib is None:
                res[name] = {"error": err}
                continue
            st, uo, po, ms = run(lib, q, delta, s0, B, T, D, args.rounds)
            res[name] = {
                "maxdiff_vs_ref": float(max(
                    np.abs(st - st_ref).max(), np.abs(uo - u_ref).max(),
                    np.abs(po - p_ref).max())),
                "median_ms": ms,
            }
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    info["shape"] = {"B": B, "T": T, "D": D}
    info["results"] = res
    a, b = res.get("avx2", {}), res.get("avx512", {})
    if "median_ms" in a and "median_ms" in b:
        info["speedup_512_vs_2"] = round(a["median_ms"] / b["median_ms"], 3)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())