"""手写 AVX2 内核 vs PyTorch 扫描: 正确性 + 加速比实测.

用法: python -X utf8 experiments/_kernel_bench.py
前置: SUNNY_W64 指向 w64devkit 根目录 (或 F:\\devtools\\w64devkit).
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.kernels import kernel_available, load_scan_kernel, TOOLCHAIN_HINT  # noqa: E402
from sunnynet.ops import logspace_scan  # noqa: E402

P = ctypes.POINTER(ctypes.c_float)


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def ref_forward(p, nu, s0):
    T = p.shape[1]
    s = s0.clone()
    outs = []
    for t in range(T):
        s = p[:, t] * s + (1 - p[:, t]) * nu[:, t]
        outs.append(s.clone())
    return torch.stack(outs, dim=1)


def main():
    if not kernel_available():
        print(TOOLCHAIN_HINT)
        return 1
    lib = load_scan_kernel()
    torch.set_num_threads(1)  # 公平: 内核是单线程

    # --- 正确性 ---
    B, T, N = 2, 128, 64
    rng = np.random.default_rng(0)
    p_np = (0.5 + 0.49 * rng.random((B, T, N))).astype(np.float32)
    nu_np = rng.standard_normal((B, T, N)).astype(np.float32)
    s0_np = np.zeros((B, N), dtype=np.float32)
    out = np.empty((B, T, N), dtype=np.float32)
    send = np.empty((B, N), dtype=np.float32)
    lib.scan_f32(p_np.ctypes.data_as(P), nu_np.ctypes.data_as(P), s0_np.ctypes.data_as(P),
                 out.ctypes.data_as(P), send.ctypes.data_as(P), B, T, N)
    ref = ref_forward(torch.from_numpy(p_np), torch.from_numpy(nu_np),
                      torch.zeros(B, N))
    fwd_diff = float((torch.from_numpy(out) - ref).abs().max())

    # 反向梯度与 autograd 对齐
    pt = torch.from_numpy(p_np).clone().requires_grad_(True)
    nt = torch.from_numpy(nu_np).clone().requires_grad_(True)
    dout = rng.standard_normal((B, T, N)).astype(np.float32)
    seq = ref_forward(pt, nt, torch.zeros(B, N))
    (seq * torch.from_numpy(dout)).sum().backward()
    dp = np.zeros((B, T, N), dtype=np.float32)
    dnu = np.zeros((B, T, N), dtype=np.float32)
    ds0 = np.zeros((B, N), dtype=np.float32)
    lib.scan_bwd_f32(p_np.ctypes.data_as(P), nu_np.ctypes.data_as(P),
                     s0_np.ctypes.data_as(P), ref.numpy().astype(np.float32).ctypes.data_as(P),
                     dout.ctypes.data_as(P), dp.ctypes.data_as(P),
                     dnu.ctypes.data_as(P), ds0.ctypes.data_as(P), B, T, N)
    dp_diff = float(np.abs(dp - pt.grad.numpy()).max())
    dnu_diff = float(np.abs(dnu - nt.grad.numpy()).max())

    print(f"正确性: 前向 maxdiff={fwd_diff:.3e}  dp={dp_diff:.3e}  dnu={dnu_diff:.3e}")
    assert fwd_diff < 1e-5 and dp_diff < 1e-4 and dnu_diff < 1e-4, "内核数值不一致"

    # --- 性能 ---
    rows = []
    for B_, T_, N_ in [(8, 256, 160), (8, 512, 160), (8, 1024, 160),
                       (8, 1024, 512), (8, 2048, 512), (4, 4096, 512)]:
        p_ = (0.6 + 0.39 * rng.random((B_, T_, N_))).astype(np.float32)
        nu_ = rng.standard_normal((B_, T_, N_)).astype(np.float32)
        s_ = np.zeros((B_, N_), dtype=np.float32)
        o_ = np.empty((B_, T_, N_), dtype=np.float32)
        e_ = np.empty((B_, N_), dtype=np.float32)

        def ck():
            lib.scan_f32(p_.ctypes.data_as(P), nu_.ctypes.data_as(P), s_.ctypes.data_as(P),
                         o_.ctypes.data_as(P), e_.ctypes.data_as(P), B_, T_, N_)

        tp = torch.from_numpy(p_)
        tn = torch.from_numpy(nu_)
        ts = torch.zeros(B_, N_)

        def tk():
            return logspace_scan(tn, tp, ts, chunk=64)

        c_ms, t_ms = bench(ck), bench(tk)
        rows.append({"B": B_, "T": T_, "N": N_, "c_ms": round(c_ms, 4),
                     "torch_ms": round(t_ms, 4), "speedup": round(t_ms / c_ms, 2)})
        print(f"  B={B_} T={T_:5d} N={N_:3d} | AVX2 {c_ms:8.3f} ms | "
              f"torch {t_ms:9.3f} ms | {t_ms / c_ms:6.2f}x")

    out = ROOT / "experiments" / "results" / "_kernel_bench.json"
    out.write_text(json.dumps({
        "correctness": {"fwd_maxdiff": fwd_diff, "dp_maxdiff": dp_diff,
                        "dnu_maxdiff": dnu_diff},
        "rows": rows,
        "note": "单线程对比 (内核单线程, torch.set_num_threads(1)); "
                "真实训练 torch 用 8 线程, 加速比会收窄",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())