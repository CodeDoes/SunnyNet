"""SunnyNet 原创基础算子. 推导见 docs/ARCHITECTURE.md."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "poly_sigmoid",
    "softwedge",
    "CenteredScaleNorm",
    "multiscale_decay",
    "logspace_scan",
]


def poly_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """s(z) = 0.5 + 0.5 * z / (1 + |z|),  值域 (0,1),  s'(z) = 0.5/(1+|z|)^2."""
    return 0.5 + 0.5 * (z / (1.0 + z.abs()))


def softwedge(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """f(z) = z + a * s(z),  a in (0,1).  f'(z) in (1, 1+a/2], 恒 > 1."""
    return z + a * poly_sigmoid(z)


class CenteredScaleNorm(nn.Module):
    """y = gamma * (x - mean + c) / sqrt(var + s_min^2 + eps)."""

    def __init__(self, dim: int, eps: float = 1e-5, s_min: float = 0.25):
        super().__init__()
        self.eps = eps
        self.s_min = s_min
        self.gamma = nn.Parameter(torch.ones(dim))
        self.center = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
        return self.gamma * (x - mu + self.center) * inv


def multiscale_decay(n: int, h_min: float = 2.0, h_max: float = 8192.0,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """逐通道遗忘上界 delta_i = 1/H_i, H_i 在 [h_min, h_max] 对数均匀."""
    if n == 1:
        h = torch.tensor([h_max], device=device, dtype=dtype)
    else:
        lo, hi = math.log(h_min), math.log(h_max)
        t = torch.arange(n, device=device, dtype=dtype) / (n - 1.0)
        h = torch.exp(lo + (hi - lo) * t)
    return 1.0 / h


def logspace_scan(nu: torch.Tensor, p: torch.Tensor, s0: torch.Tensor,
                  chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """一阶线性递推 s_t = p_t*s_{t-1} + (1-p_t)*nu_t 的闭式解.

    记 P_j = sum_{k<=j} ln p_k (inclusive), g_k = (1-p_k)*nu_k:
        s_j = exp(P_j) * ( s_in + sum_{k<=j} g_k * exp(-P_k) )
    两次 inclusive cumsum 得到整块状态, 块间串行 T/chunk 次.

    nu, p: (B,T,N);  s0: (B,N) -> (states (B,T,N), s_T (B,N))
    """
    B, T, N = nu.shape
    lp = torch.log(p)
    g = (1.0 - p) * nu
    chunk = _safe_chunk(p.detach().amin().item(), chunk)
    s = s0
    outs = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        P = torch.cumsum(lp[:, start:end], dim=1)
        A = torch.cumsum(g[:, start:end] * torch.exp(-P), dim=1)
        s_c = torch.exp(P) * (s.unsqueeze(1) + A)
        outs.append(s_c)
        s = s_c[:, -1]
    return torch.cat(outs, dim=1), s


def _safe_chunk(p_min: float, requested: int, headroom_bits: int = 48) -> int:
    """按 p 最小值限制块长, 使 exp(-P) <= 2^48 不溢出. 取值与块长无关."""
    if p_min <= 0 or p_min >= 1:
        return min(requested, 64)
    per_bit = -math.log(p_min)
    if per_bit <= 0:
        return requested
    return max(1, min(requested, int(headroom_bits * math.log(2.0) / per_bit)))