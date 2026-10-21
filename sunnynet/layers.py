"""SunnyNet 网络层. 块内数据流与推导见 docs/ARCHITECTURE.md 第 1 节.

状态协议 SunnyState(s=(B,L,D), hist=(B,L,k-1,D)) 保证单步与整段严格一致.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import orthogonal_spectrum_
from .kernels.core_auto import core_forward, dw_step, kernel_active, step_core
from .ops import CenteredScaleNorm, horizon_logit, logspace_scan

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
        self.use_residual = ab.get("residual", True)
        self.use_state_norm = ab.get("state_norm", False)
        self.use_state_mul = ab.get("state_mul", False)

        self.norm_in = CenteredScaleNorm(dim)
        self.norm_mix = CenteredScaleNorm(inner)
        self.norm_s = CenteredScaleNorm(dim) if self.use_state_norm else None
        self.W_g = nn.Linear(inner, dim, bias=False) if self.use_state_mul else None
        self.circ = StateCirculant(dim) if self.use_circulant else None
        self.dw = DepthwiseCausal(dim, taps) if (self.use_local_taps and taps > 1) else None
        self.V = nn.Linear(dim, rank, bias=False)
        self.U = nn.Linear(rank, 2 * dim, bias=False)
        self.W_z = nn.Linear(dim, inner, bias=False)
        self.W_c = nn.Linear(inner, dim, bias=False)
        self.W_s = nn.Linear(dim, dim, bias=False)

        self.a = nn.Parameter(torch.tensor(0.5))
        self.b_gain = nn.Parameter(torch.tensor(0.5))
        self.h_min, self.h_max = h_min, h_max
        self.register_buffer("h_logit0", horizon_logit(dim), persistent=True)
        self.register_buffer("delta_max", torch.tensor(1.0 / h_min), persistent=True)
        self.register_buffer("delta_min", torch.tensor(1.0 / h_max), persistent=True)
        self.theta = nn.Parameter(self.h_logit0.clone())
        self.p_floor = 1.0 / h_max
        self.p_min_bound = float((1.0 - 1.0 / h_min))
        # p = 1 - delta*(1-sigmoid(gate)), delta<=1/h_min, 故 p>=p_min_bound; 上界恒 <1.
        # 解析下界已高于 p_floor 时 clamp 不可达, 构造期判掉省一次全宽张量往返.
        self._need_clamp = self.p_min_bound < self.p_floor
        self.group = _pick_group(dim, group)
        self._w_t: torch.Tensor | None = None
        self._w_t_ver = -1
        self._delta_cache: torch.Tensor | None = None
        self._delta_ver = -1
        self._init_spectral(rho0, decay_beta, layer_index, rank, generator)

    def delta(self) -> torch.Tensor:
        """逐通道遗忘上界 delta_i = 1/H_i, H 在对数刻度上扫 h_min~h_max."""
        span = math.log(self.h_max / self.h_min)
        H = self.h_min * torch.exp(span * torch.sigmoid(self.theta))
        return 1.0 / H

    def delta_cached(self) -> torch.Tensor:
        """推理态用: delta 只依赖 theta, 每步重算是纯浪费. theta 变更时重算."""
        if self._delta_cache is None or self._delta_ver != self.theta._version:
            self._delta_cache = self.delta().detach()
            self._delta_ver = self.theta._version
        return self._delta_cache

    def _init_spectral(self, rho0, beta, layer_index, rank, generator):
        rho = rho0 * (float(layer_index + 1) ** -beta)
        with torch.no_grad():
            orthogonal_spectrum_(self.V.weight, rho, generator)
            orthogonal_spectrum_(self.U.weight, rho, generator)
            orthogonal_spectrum_(self.W_z.weight, rho, generator)
            orthogonal_spectrum_(self.W_c.weight, rho, generator)
            orthogonal_spectrum_(self.W_s.weight, rho * 0.5, generator)

    def _use_kernel(self, ap: torch.Tensor) -> bool:
        """内核前提: fp32 CPU 连续, group==8, 非 ablated 门控, 未 clamp."""
        if not kernel_active():
            return False
        if self.group != 8 or self.dim % 8 or not self.use_decay_gate or self._need_clamp:
            return False
        if ap.dtype != torch.float32 or not ap.is_cpu or not ap.is_contiguous():
            return False
        return ap.shape[-1] == self.dim and self.use_circulant

    def _core_kernel(self, ap: torch.Tensor, s: torch.Tensor):
        """融合 core: q -> (u, states). 复用 torch 侧 q 计算与 delta。"""
        q = self.U(self.V(ap))
        delta = self.delta().contiguous()
        (u, states), = (core_forward(q, delta, s),)
        return u, states

    def _compete(self, gate: torch.Tensor) -> torch.Tensor:
        """组内成对竞争门控 w = sigmoid(g_i-g_j)*sigmoid(g_i-g_k) in (0,1). Jacobian O(1)."""
        B, T, D = gate.shape
        gg = gate.view(B, T, D // self.group, self.group)
        w = torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
            torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))
        return w.reshape(B, T, D)

    def _apply_local(self, raw: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """卷积历史 prepend 到序列前端. hist 全零时免 cat (fp64 逐位相等)."""
        if self.dw is None:
            return raw
        k = self.taps - 1
        if k and hist.abs().max().item() != 0.0:
            rf = torch.cat([hist, raw], dim=1)
            return (rf + self.dw(rf))[:, k:]
        return raw + self.dw(raw)

    def _core(self, a: torch.Tensor):
        """扫描前的输入侧计算, 与滚动状态无关."""
        q = self.U(self.V(a))
        u_raw, gate = q.split(self.dim, dim=-1)
        u = u_raw * self._compete(gate)
        delta = self.delta().view(1, 1, -1)
        if self.use_decay_gate:
            p = 1.0 - delta * (1.0 - torch.sigmoid(gate))
        else:
            p = (1.0 - delta).expand_as(u)
        if self._need_clamp:
            p = p.clamp(self.p_floor, 1.0)
        return u, gate, p

    def _prev(self, states: torch.Tensor, s_in: torch.Tensor) -> torch.Tensor:
        """位置 0 用 s_in, 其余用前一步状态."""
        return torch.cat([s_in[:, None, :], states[:, :-1]], dim=1)

    def _feedback(self, states: torch.Tensor, s_in: torch.Tensor) -> torch.Tensor:
        """状态反馈: f_t = circ(S_{t-1}), 位置 0 用 circ(s_in)."""
        prev = self._prev(states, s_in)
        if self.circ is None:
            return prev
        return self.circ(prev)

    def _Ws_eff(self) -> torch.Tensor:
        """把 circ 折进 W_s: W_s(circ(x)) == x @ W_eff^T (fp64 相对误差 1e-15).

        circ 与 W_s 都只作用在特征维, 可交换为单次矩阵乘; 省掉整条 circ 路径.
        """
        W = self.W_s.weight
        t0, t1, t2 = self.circ.t[0], self.circ.t[1], self.circ.t[2]
        return t0 * torch.roll(W, -1, dims=1) + t1 * W + t2 * torch.roll(W, 1, dims=1)

    def _readout(self, u: torch.Tensor, s_prev: torch.Tensor):
        """输出侧非线性支路, 前向与单步共用."""
        if self.use_current:
            return self.norm_mix(self._wedge(self.W_z(u)))
        return self.norm_mix(self.W_z(u))

    def _wedge(self, z: torch.Tensor) -> torch.Tensor:
        """softwedge 去掉常数项: z + a*s(z) = z*(1 + a/2/(1+|z|)) + a/2.

        常数 a/2 在紧邻的 CenteredScaleNorm 中被中心化吸收 (fp64 相对误差 3e-16),
        因此省一次最宽张量上的加法.
        """
        a = self.a.tanh()
        return z * (1.0 + 0.5 * a / (1.0 + z.abs()))

    def _emit(self, u: torch.Tensor, states: torch.Tensor, s_in: torch.Tensor):
        y = self._readout(u, None)
        sn = self.norm_s(states) if self.norm_s is not None else states
        prev = self._prev(states, s_in)
        if self.circ is not None:
            sfb = F.linear(prev, self._Ws_eff())
        else:
            sfb = self.W_s(prev)
        core = self.W_c(y) + self.b_gain * sn + sfb
        if self.use_state_mul:
            core = core + self.W_g(y) * sn
        return core

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
        if not pad and self._use_kernel(ap):
            u, states = self._core_kernel(ap, s)
        else:
            u, gate, p = self._core(ap)
            states, _ = logspace_scan(u, p, s, chunk=chunk, p_min_hint=self.p_min_bound)
        out = self._emit(u, states, s)
        if pad:
            out = out[:, :T]
            states = states[:, :T]
        if self.use_residual:
            out = out + x
        return out, states[:, -1], hist_new

    def step(self, x_t: torch.Tensor, s: torch.Tensor, hist: torch.Tensor):
        """单步: x_t (B,D) -> (out (B,D), s' (B,D), hist_new (B,k-1,D))."""
        raw = self.norm_in(x_t)
        k = max(0, self.taps - 1)
        use_k = (kernel_active() and self.group == 8 and not self._need_clamp
                 and self.use_circulant and self.use_decay_gate and k > 0
                 and raw.dtype == torch.float32 and raw.is_contiguous()
                 and s.is_contiguous() and hist.is_contiguous())
        if k:
            if use_k:
                if self._w_t is None or self._w_t_ver != self.dw.weight._version:
                    self._w_t = self.dw.weight.detach().transpose(1, 2) \
                        .reshape(self.taps, self.dim).contiguous()
                    self._w_t_ver = self.dw.weight._version
                a_t = raw + dw_step(raw, hist, self._w_t, self.taps)
            else:
                a_t = raw + self.dw.step(raw, hist)
            hist_new = torch.cat([raw[:, None, :], hist[:, :k - 1]], dim=1)
        else:
            a_t = raw
            hist_new = hist
        if use_k:
            q = self.U(self.V(a_t))
            u_t, s_new = step_core(q, self.delta_cached(), s)
        else:
            u_t, gate, p_t = self._core(a_t[:, None, :])
            u_t, gate, p_t = u_t[:, 0], gate[:, 0], p_t[:, 0]
            s_new = p_t * s + (1.0 - p_t) * u_t
        y = self._readout(u_t[:, None, :], None)[:, 0]
        sn = self.norm_s(s_new[:, None, :])[:, 0] if self.norm_s is not None else s_new
        if self.circ is not None:
            sfb = F.linear(s, self._Ws_eff())
        else:
            sfb = self.W_s(s)
        out = self.W_c(y) + self.b_gain * sn + sfb
        if self.use_state_mul:
            out = out + self.W_g(y) * sn
        if self.use_residual:
            out = out + x_t
        return out, s_new, hist_new