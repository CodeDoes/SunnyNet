"""SunnyNet 网络层. 块内数据流与推导见 docs/ARCHITECTURE.md 第 1 节.

状态协议 SunnyState(s=(B,L,D), hist=(B,L,k-1,D)) 保证单步与整段严格一致.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import orthogonal_spectrum_
from .ops import CenteredScaleNorm, logspace_scan, multiscale_decay, softwedge

__all__ = ["StateCirculant", "DepthwiseCausal", "SunnyBlock", "SunnyState"]


@dataclass
class SunnyState:
    """跨块/跨层的完整状态: 逐块滚动状态 + 逐块因果卷积历史."""

    s: torch.Tensor
    hist: torch.Tensor

    def to(self, device=None, dtype=None):
        return SunnyState(self.s.to(device=device, dtype=dtype),
                          self.hist.to(device=device, dtype=dtype))


class StateCirculant(nn.Module):
    """状态 3 抽头循环滤波 y = t0*roll(s,1) + t1*s + t2*roll(s,-1). 无矩阵乘法."""

    def __init__(self, dim: int):
        super().__init__()
        self.t = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a, b, c = self.t[0], self.t[1], self.t[2]
        return a * torch.roll(s, 1, dims=-1) + b * s + c * torch.roll(s, -1, dims=-1)


class DepthwiseCausal(nn.Module):
    """深度可分因果卷积, 每通道独立 k 抽头, 参数量 O(C*k).

    前向用 left-padding; 单步推理用显式历史缓冲, 两者严格一致.
    """

    def __init__(self, channels: int, taps: int = 4):
        super().__init__()
        self.taps = taps
        self.weight = nn.Parameter(torch.randn(channels, 1, taps) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.taps <= 1:
            return x * self.weight[:, 0, 0].view(1, 1, C)
        xt = F.pad(x.transpose(1, 2), (self.taps - 1, 0))
        return F.conv1d(xt, self.weight, groups=C).transpose(1, 2)

    def step(self, x_t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """x_t (B,C), hist (B, taps-1, C), 其中 hist[:,0] 是上一时刻输入."""
        C = x_t.shape[-1]
        if self.taps <= 1:
            return x_t * self.weight[:, 0, 0].view(1, C)
        acc = x_t * self.weight[:, 0, self.taps - 1].view(1, C)
        for k in range(self.taps - 1):
            acc = acc + hist[:, k] * self.weight[:, 0, self.taps - 2 - k].view(1, C)
        return acc


def _pick_group(dim: int, requested: int) -> int:
    """选一个能整除 dim 的组大小, 且至少为 3 (需要两个组内邻居)."""
    for g in (requested, 4, 3):
        if g >= 3 and dim % g == 0:
            return g
    for g in range(3, dim + 1):
        if dim % g == 0:
            return g
    return dim


class SunnyBlock(nn.Module):
    """SunnyNet 单块."""

    def __init__(self, dim: int, inner: int, rank: int = 128, taps: int = 4,
                 layer_index: int = 0, rho0: float = 1.0, decay_beta: float = 0.0,
                 h_min: float = 2.0, h_max: float = 8192.0, scan_chunk: int = 64,
                 group: int = 8, ablation: dict | None = None, generator=None):
        super().__init__()
        ab = ablation or {}
        self.dim = dim
        self.inner = inner
        self.taps = taps
        self.scan_chunk = scan_chunk
        self.use_decay_gate = ab.get("decay_gate", True)
        self.use_circulant = ab.get("circulant", True)
        self.use_local_taps = ab.get("local_taps", True)
        self.use_current = ab.get("current", True)

        self.norm_in = CenteredScaleNorm(dim)
        self.norm_mix = CenteredScaleNorm(inner)
        self.circ = StateCirculant(dim) if self.use_circulant else None
        self.dw = DepthwiseCausal(dim, taps) if (self.use_local_taps and taps > 1) else None
        self.V = nn.Linear(dim, rank, bias=False)
        self.U = nn.Linear(rank, 2 * dim, bias=False)
        self.W_z = nn.Linear(dim, inner, bias=False)
        self.W_c = nn.Linear(inner, dim, bias=False)
        self.W_s = nn.Linear(dim, dim, bias=False)

        self.a = nn.Parameter(torch.tensor(0.5))
        self.b_gain = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("delta", multiscale_decay(dim, h_min, h_max), persistent=True)
        self.p_floor = 1.0 / h_max
        self.group = _pick_group(dim, group)
        self._init_spectral(rho0, decay_beta, layer_index, rank, generator)

    def _init_spectral(self, rho0, beta, layer_index, rank, generator):
        rho = rho0 * (float(layer_index + 1) ** -beta)
        with torch.no_grad():
            orthogonal_spectrum_(self.V.weight, rho, generator)
            orthogonal_spectrum_(self.U.weight, rho, generator)
            orthogonal_spectrum_(self.W_z.weight, rho, generator)
            orthogonal_spectrum_(self.W_c.weight, rho, generator)
            orthogonal_spectrum_(self.W_s.weight, rho * 0.5, generator)

    def _compete(self, gate: torch.Tensor) -> torch.Tensor:
        """组内成对竞争门控 w = sigmoid(g_i-g_j)*sigmoid(g_i-g_k) in (0,1). Jacobian O(1)."""
        B, T, D = gate.shape
        gg = gate.view(B, T, D // self.group, self.group)
        w = torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
            torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))
        return w.reshape(B, T, D)

    def _apply_local(self, raw: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """卷积历史 prepend 到序列前端."""
        if self.dw is None:
            return raw
        k = self.taps - 1
        rf = torch.cat([hist, raw], dim=1) if k else raw
        return (rf + self.dw(rf))[:, k:]

    def _core(self, a: torch.Tensor):
        """扫描前的输入侧计算, 与滚动状态无关."""
        q = self.U(self.V(a))
        u_raw, gate = q.split(self.dim, dim=-1)
        u = u_raw * self._compete(gate)
        if self.use_decay_gate:
            p = 1.0 - self.delta.view(1, 1, -1) * (1.0 - torch.sigmoid(gate))
        else:
            p = (1.0 - self.delta).view(1, 1, -1).expand_as(u)
        return u, gate, p.clamp(self.p_floor, 1.0)

    def _feedback(self, states: torch.Tensor, s_in: torch.Tensor) -> torch.Tensor:
        """状态反馈: f_t = circ(S_{t-1}), 位置 0 用 circ(s_in)."""
        if self.circ is None:
            return torch.cat([s_in[:, None, :], states[:, :-1]], dim=1)
        prev = torch.cat([s_in[:, None, :], states[:, :-1]], dim=1)
        return self.circ(prev)

    def _emit(self, u: torch.Tensor, states: torch.Tensor, s_in: torch.Tensor):
        if self.use_current:
            y = self.norm_mix(softwedge(self.W_z(u), self.a.tanh()))
        else:
            y = self.norm_mix(self.W_z(u))
        return self.W_c(y) + self.b_gain * states + self.W_s(self._feedback(states, s_in))

    def forward(self, x: torch.Tensor, s: torch.Tensor, hist: torch.Tensor):
        """x (B,T,D) -> (out, s_T, hist_new).

        T 会 pad 到整除 chunk, 返回状态取自位置 T-1 而非 padding 位置.
        """
        B, T, D = x.shape
        raw = self.norm_in(x)
        a = self._apply_local(raw, hist)
        k = max(0, self.taps - 1)
        if k:
            hist_new = raw[:, T - k:T] if T >= k else \
                torch.cat([raw.new_zeros(B, k - T, D), raw], dim=1)
        else:
            hist_new = hist

        chunk = min(self.scan_chunk, T)
        pad = (-T) % chunk
        ap = torch.cat([a, a.new_zeros(B, pad, D)], dim=1) if pad else a
        u, gate, p = self._core(ap)
        states, _ = logspace_scan(u, p, s, chunk=chunk)
        out = self._emit(u, states, s)
        if pad:
            out = out[:, :T]
            states = states[:, :T]
        return out, states[:, -1], hist_new

    def step(self, x_t: torch.Tensor, s: torch.Tensor, hist: torch.Tensor):
        """单步: x_t (B,D) -> (out (B,D), s' (B,D), hist_new (B,k-1,D))."""
        raw = self.norm_in(x_t)
        k = max(0, self.taps - 1)
        a_t = raw if k == 0 else raw + self.dw.step(raw, hist)
        if k:
            hist_new = torch.cat([raw[:, None, :], hist[:, :k - 1]], dim=1)
        else:
            hist_new = hist
        u_t, gate, p_t = self._core(a_t[:, None, :])
        u_t, gate, p_t = u_t[:, 0], gate[:, 0], p_t[:, 0]
        s_new = p_t * s + (1.0 - p_t) * u_t
        prev = self.circ(s) if self.circ is not None else s
        if self.use_current:
            y = self.norm_mix(softwedge(self.W_z(u_t), self.a.tanh()))
        else:
            y = self.norm_mix(self.W_z(u_t))
        out = self.W_c(y) + self.b_gain * s_new + self.W_s(prev)
        return out, s_new, hist_new