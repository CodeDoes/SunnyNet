"""融合 core 内核的 autograd 包装 + 全局开关.

q = U(V(a)) 在 torch 侧算 (矩阵乘走 MKL), 内核只接手它后面的
竞争门控 + 门控概率 + 扫描整段, 输出 (u, states)。
delta, s0 为内核输入, 梯度经内核反向回传后由 torch autograd continuing。
"""
from __future__ import annotations

import ctypes
import os

import torch

from . import kernel_available, load_kernel, set_kernel_threads

__all__ = ["CoreAVX2", "core_forward", "step_core", "dw_step",
           "kernel_active", "set_kernel_active"]

_LIB = None
_ACTIVE = os.environ.get("SUNNY_KERNEL", "0") == "1"


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = load_kernel()
    return _LIB


def set_kernel_active(flag: bool = True, threads: int | None = None) -> bool:
    """打开/关闭内核路径. 返回实际是否处于可用状态."""
    global _ACTIVE
    _ACTIVE = bool(flag)
    if _ACTIVE and threads:
        set_kernel_threads(threads)
    return kernel_active()


def kernel_active() -> bool:
    return _ACTIVE and kernel_available()


class CoreAVX2(torch.autograd.Function):
    """(u_raw, gate) 打包成 q (B,T,2D) + delta (D,) + s0 (B,D) -> (u, states)。"""

    @staticmethod
    def forward(ctx, q, delta, s0):
        B, T, D2 = q.shape
        D = D2 // 2
        q_c = q.contiguous().float()
        delta_c = delta.contiguous().float()
        s0_c = s0.contiguous().float()
        states = torch.empty(B, T, D, dtype=torch.float32)
        u = torch.empty(B, T, D, dtype=torch.float32)
        p = torch.empty(B, T, D, dtype=torch.float32)
        _lib().core_fwd_f32(q_c.data_ptr(), delta_c.data_ptr(), s0_c.data_ptr(),
                            states.data_ptr(), u.data_ptr(), p.data_ptr(), B, T, D)
        ctx.save_for_backward(q_c, delta_c, s0_c, states, u, p)
        return u, states

    @staticmethod
    def backward(ctx, du, dstate):
        q_c, delta_c, s0_c, states, u, p = ctx.saved_tensors
        B, T, D2 = q_c.shape
        D = D2 // 2
        dq = torch.empty_like(q_c)
        ddelta = torch.zeros(D, dtype=torch.float32)
        ds0 = torch.empty_like(s0_c)
        _lib().core_bwd_f32(
            q_c.data_ptr(), delta_c.data_ptr(), s0_c.data_ptr(),
            states.data_ptr(), u.data_ptr(), p.data_ptr(),
            dstate.contiguous().float().data_ptr(), du.contiguous().float().data_ptr(),
            dq.data_ptr(), ddelta.data_ptr(), ds0.data_ptr(), B, T, D)
        return dq, ddelta, ds0


def core_forward(q: torch.Tensor, delta: torch.Tensor, s0: torch.Tensor):
    """走内核; 调用方须先用 block._use_kernel() 确认前提。"""
    return CoreAVX2.apply(q, delta, s0)


def step_core(q: torch.Tensor, delta: torch.Tensor, s: torch.Tensor):
    """推理态单步: q (B,2D) + delta (D,) + s (B,D) -> (u_t, s_new).

    data_ptr() 直接给整数, 比 data_as(POINTER) 少一次 ctypes 对象构造.
    contiguous/float 前置判掉, 命中时不做拷贝 (解码每步都要走这里).
    """
    lib = _lib()
    B, D2 = q.shape
    D = D2 // 2
    if q.dtype != torch.float32 or not q.is_contiguous():
        q = q.contiguous().float()
    if s.dtype != torch.float32 or not s.is_contiguous():
        s = s.contiguous().float()
    if delta.dtype != torch.float32 or not delta.is_contiguous():
        delta = delta.contiguous().float()
    s_new = torch.empty_like(s)
    u = torch.empty(B, D, dtype=torch.float32)
    lib.step_core_f32(q.data_ptr(), delta.data_ptr(), s.data_ptr(),
                      s_new.data_ptr(), u.data_ptr(), B, D)
    return u, s_new


def dw_step(x: torch.Tensor, hist: torch.Tensor, w_t: torch.Tensor, taps: int):
    """深度可分因果卷积单步: x (B,D) + hist (B,taps-1,D) + w (taps,D) -> acc (B,D)."""
    lib = _lib()
    B, D = x.shape
    if x.dtype != torch.float32 or not x.is_contiguous():
        x = x.contiguous().float()
    if hist.dtype != torch.float32 or not hist.is_contiguous():
        hist = hist.contiguous().float()
    acc = torch.empty_like(x)
    lib.dw_step_f32(x.data_ptr(), hist.data_ptr(), w_t.data_ptr(),
                    acc.data_ptr(), B, D, taps)
    return acc