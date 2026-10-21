"""AVX2 scan 内核的 autograd 包装.

与 sunnynet.ops.logspace_scan 数值等价 (前向 <1e-5, 反向 <1e-4)。
未编译/不可用时由调用方回退到 torch 实现, 见 scan_auto。
"""
from __future__ import annotations

import ctypes

import torch

from . import kernel_available, load_scan_kernel

__all__ = ["ScanAVX2", "scan_auto"]

_LIB = None
_P = ctypes.POINTER(ctypes.c_float)


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = load_scan_kernel()
    return _LIB


class ScanAVX2(torch.autograd.Function):
    """s_t = p_t*s_{t-1} + (1-p_t)*u_t 的前向/反向, 走 C 内核。

    只接受 fp32 连续张量; 形状 (B,T,N) 与 (B,N)。
    """

    @staticmethod
    def forward(ctx, u, p, s0):
        B, T, N = u.shape
        u_c = u.contiguous().float()
        p_c = p.contiguous().float()
        s0_c = s0.contiguous().float()
        out = torch.empty_like(u_c)
        send = torch.empty_like(s0_c)
        _lib().scan_f32(
            p_c.data_ptr(), u_c.data_ptr(), s0_c.data_ptr(),
            out.data_ptr(), send.data_ptr(),
            B, T, N)
        ctx.save_for_backward(u_c, p_c, s0_c, out)
        return out

    @staticmethod
    def backward(ctx, dout):
        u_c, p_c, s0_c, out = ctx.saved_tensors
        B, T, N = u_c.shape
        dout_c = dout.contiguous().float()
        dp = torch.zeros_like(p_c)
        du = torch.zeros_like(u_c)
        ds0 = torch.zeros_like(s0_c)
        _lib().scan_bwd_f32(
            p_c.data_ptr(), u_c.data_ptr(), s0_c.data_ptr(), out.data_ptr(),
            dout_c.data_ptr(), dp.data_ptr(), du.data_ptr(), ds0.data_ptr(),
            B, T, N)
        return du, dp, ds0


def scan_auto(u: torch.Tensor, p: torch.Tensor, s0: torch.Tensor,
              chunk: int = 64, p_min_hint: float | None = None):
    """内核可用且输入为 fp32/CUDA-free 时走 AVX2, 否则回退 torch 实现。"""
    from ..ops import logspace_scan

    if (kernel_available() and u.is_cpu and u.dtype == torch.float32
            and p.dtype == torch.float32 and u.is_contiguous()
            and p.is_contiguous() and u.shape == p.shape):
        states = ScanAVX2.apply(u, p, s0)
        return states, states[:, -1]
    return logspace_scan(u, p, s0, chunk=chunk, p_min_hint=p_min_hint)