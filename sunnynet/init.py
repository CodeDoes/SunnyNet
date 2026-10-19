"""谱锚定初始化: 直接控制权重奇异值谱, 不用 fan_in/fan_out 方差公式."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["orthogonal_spectrum_", "spectral_anchored_init"]


def orthogonal_spectrum_(weight: torch.Tensor, rho: float,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """原地写入谱半径为 rho 的正交切片, 条件数为 1."""
    out_f, in_f = weight.shape
    n = max(out_f, in_f)
    a = torch.randn(n, n, generator=generator, dtype=weight.dtype, device=weight.device)
    q, r = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    base = q[:out_f, :in_f] if out_f <= in_f else q[:, :in_f]
    with torch.no_grad():
        weight.copy_(base.mul(rho))
    return weight


def spectral_anchored_init(module: nn.Module, rho0: float = 1.0,
                           beta: float = 0.0, depth_index: int = 0,
                           generator: torch.Generator | None = None) -> None:
    """对模块内矩阵参数施加谱锚定初始化.

    beta 保持 0: 实测按层深衰减 rho_l = rho0*(l+1)^-beta 会使 24 层梯度降到 1e-7 量级.
    """
    rho = rho0 * (float(depth_index + 1) ** -beta)
    with torch.no_grad():
        for p in module.parameters(recurse=False):
            if p.dim() == 2:
                orthogonal_spectrum_(p.data, rho, generator=generator)
            elif p.dim() == 1:
                p.data.fill_(1.0)