"""O(N) 序列架构对照基线.

同属线性复杂度的公开架构, 仅作公平对照, 不属于 SunnyNet 本体.
全部手写实现以精确控制参数量 (不调用第三方实现).

包含:
  LinearAttention  线性注意力 (Katharopoulos 2020 形式): 特征映射 phi, 无 softmax
  RWKV             RWKV-4 风格: 时间混合 wkv + 通道混合
  Mamba            Mamba/S6 风格: 输入相关选择性门控 + 深度卷积 + SiLU
  Hyena            Hyena 风格: 长卷积 + 门控 + 深度卷积
  H3               H3-2 风格: 对角 SSM + 短卷积 + 乘性门控

规模对齐: 各架构层数/宽度按计算栈参数量与 SunnyNet 对齐, 由 solve_linear_baseline 求解.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import gated_scan

__all__ = [
    "LinearConfig",
    "LinearAttentionLM",
    "RWKVLM",
    "MambaLM",
    "HyenaLM",
    "H3LM",
    "build_linear_baseline",
    "LINEAR_ARCHS",
]

LINEAR_ARCHS = ("linear_attn", "rwkv", "mamba", "hyena", "h3")
_ARCH_LM = {}


@dataclass
class LinearConfig:
    arch: str = "linear_attn"
    vocab_size: int = 8192
    d_model: int = 128
    n_layers: int = 4
    d_ff: int = 256
    max_len: int = 2048
    tie_embeddings: bool = True
    fft_conv: bool = False


class _LMBase(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        self.cfg = cfg
        self.arch = cfg.arch
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList()
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    def _finish_init(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(max(1, 2 * self.cfg.n_layers)))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.head.weight.numel()
        return n

    def forward(self, idx, state=None, return_state=False):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if return_state:
            return logits, None
        return logits


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LinearAttentionBlock(nn.Module):
    """线性注意力: phi(x)=(elu(x)+1), 因果累积 K^T V, 无 softmax, O(N)."""

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = max(1, d // 32)
        self.d_head = d // self.n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = _FFN(d, cfg.d_ff)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)
        kv_cum = torch.cumsum(kv, dim=1)
        z = torch.cumsum(k, dim=1)
        num = torch.einsum("bhtd,bhtde->bhte", q, kv_cum)
        den = torch.einsum("bhtd,bhtd->bht", q, z).unsqueeze(-1) + 1e-6
        y = (num / den).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))


class RWKVBlock(nn.Module):
    """RWKV-4 风格: wkv 用闭式扫描实现, 通道混合用 relu^2."""

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.decay = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.r_proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=True)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=True)
        self.receptance = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.decay.weight.zero_()
            self.decay.weight += torch.eye(d) * 0.5

    def forward(self, x):
        h = self.ln1(x)
        p = torch.sigmoid(self.decay(h)).clamp(1e-4, 1.0 - 1e-6)
        k = self.k_proj(h)
        v = self.v_proj(h)
        states, _ = gated_scan(p, k * v, torch.zeros_like(k[:, 0]))
        r = torch.sigmoid(self.r_proj(h))
        y = self.out(r * states)
        x = x + y
        h2 = self.ln2(x)
        return x + self.value(F.relu(self.key(h2)) ** 2)


class MambaBlock(nn.Module):
    """Mamba/S6 风格: 输入相关选择性门控 + 因果深度卷积 + SiLU 门."""

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d = cfg.d_model
        d_in = 2 * d
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, d_in, bias=False)
        self.conv = nn.Conv1d(d_in, d_in, kernel_size=4, groups=d_in,
                              padding=3, bias=True)
        dt_rank = max(1, d // 8)
        self.dt_proj = nn.Linear(dt_rank, d, bias=True)
        self.x_proj = nn.Linear(d, dt_rank + 2 * d, bias=False)
        self.A_log = nn.Parameter(torch.zeros(d))
        self.D = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ff = _FFN(d, cfg.d_ff)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        xz = self.in_proj(h)
        xz = self.conv(xz.transpose(1, 2))[..., :T].transpose(1, 2)
        xs, z = xz.split(C, dim=-1)
        xs = F.silu(xs)
        dbc = self.x_proj(xs)
        dt_r, b_, c_ = dbc.split([self.dt_proj.in_features, C, C], dim=-1)
        dt = F.softplus(self.dt_proj(dt_r))
        A = -torch.exp(self.A_log)
        decay = torch.exp(dt * A.view(1, 1, -1))                     # (B,T,C)
        bu = b_ * xs
        states, _ = gated_scan(decay, (1.0 - decay) * bu, torch.zeros_like(bu[:, 0]))
        y = states * F.silu(z) + self.D.view(1, 1, -1) * xs
        x = x + self.out_proj(y)
        return x + self.ff(self.ln2(x))


class HyenaBlock(nn.Module):
    """Hyena 风格: 长卷积(FFT) + 门控 + 深度卷积, 全 O(N log N)."""

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d = cfg.d_model
        self.use_fft = cfg.fft_conv
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.filters = nn.Parameter(torch.randn(3, d, 64) * 0.05)
        self.short = nn.Conv1d(d, d, kernel_size=3, groups=d,
                               padding=2, bias=True)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ff = _FFN(d, cfg.d_ff)

    def _long_conv(self, v, filt):
        B, T, C = v.shape
        f = filt.unsqueeze(0).expand(B, -1, -1)
        n = 1
        while n < T + f.shape[-1]:
            n *= 2
        V = torch.fft.rfft(v.transpose(1, 2), n=n, dim=-1)
        Fk = torch.fft.rfft(f, n=n, dim=-1)
        y = torch.fft.irfft(V * Fk, n=n, dim=-1)[..., :T]
        return y.transpose(1, 2)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        u, g1, g2 = self.in_proj(h).split(C, dim=-1)
        y = self._long_conv(u, self.filters[0])
        y = y * g1
        y = self._long_conv(y, self.filters[1])
        y = y * g2
        y = self._long_conv(y, self.filters[2])
        y = self.short(y.transpose(1, 2))[..., :T].transpose(1, 2)
        x = x + self.out_proj(y)
        return x + self.ff(self.ln2(x))


class H3Block(nn.Module):
    """H3-2 风格: 对角 SSM (闭式扫描) + 短卷积 + 乘性门控."""

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.short_conv = nn.Conv1d(3 * d, 3 * d, kernel_size=3, groups=3 * d,
                                    padding=2, bias=True)
        self.a_log = nn.Parameter(torch.zeros(d))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.ff = _FFN(d, cfg.d_ff)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        u, v, g = self.in_proj(h).split(C, dim=-1)
        conv = self.short_conv(torch.cat([u, v, g], dim=-1).transpose(1, 2))
        conv = conv[..., :T].transpose(1, 2)
        u, v, g = conv.split(C, dim=-1)
        decay = torch.sigmoid(-self.a_log).view(1, 1, -1).expand(B, T, C)
        states, _ = gated_scan(decay, (1.0 - decay) * v, torch.zeros_like(v[:, 0]))
        y = u * states * torch.sigmoid(g)
        x = x + self.out_proj(y)
        return x + self.ff(self.ln2(x))


BLOCKS = {
    "linear_attn": LinearAttentionBlock,
    "rwkv": RWKVBlock,
    "mamba": MambaBlock,
    "hyena": HyenaBlock,
    "h3": H3Block,
}


def _stack_count(arch: str, d: int, d_ff: int, n_layers: int) -> int:
    """计算栈参数量 (不含嵌入与 head), 用于与 SunnyNet 对齐.

    在 meta device 上实例化单块并计数: 形状真实、不占内存, 避免手推公式漏项.
    """
    cfg = LinearConfig(arch=arch, d_model=d, n_layers=1, d_ff=d_ff)
    with torch.device("meta"):
        blk = BLOCKS[arch](cfg)
    per = sum(p.numel() for p in blk.parameters())
    return per * n_layers + 2 * d


def build_linear_baseline(arch: str, target_stack: int, vocab_size: int,
                          max_len: int = 2048, n_layers: int = 6,
                          d_ff_ratio: float = 2.0):
    """枚举 (d_model, n_layers) 使计算栈参数量最接近目标, 返回 (cfg, 栈参数量, 相对偏差).

    小规模下单纯调 d 的粒度不够, 同时枚举层数可把偏差压到 1% 以内.
    """
    best = None
    step = 8
    layer_choices = [max(1, n_layers + d) for d in (-1, 0, 1)]
    for nl in layer_choices:
        for d in range(64, 2048, step):
            if arch in ("linear_attn", "rwkv") and d % 16:
                continue
            d_ff = max(step, int(d * d_ff_ratio) // step * step)
            n = _stack_count(arch, d, d_ff, nl)
            if n > target_stack * 1.5:
                break
            rel = abs(n - target_stack) / max(1, target_stack)
            key = (rel, -min(n, target_stack))
            if best is None or key < best[0]:
                best = (key, n, rel, d, d_ff, nl)
    if best is None or best[2] > 0.10:
        # 层数受限时放宽到 ±2 层, 仍优先小偏差
        for nl in sorted({max(1, n_layers + d) for d in (-2, 2)}):
            for d in range(64, 2048, step):
                if arch in ("linear_attn", "rwkv") and d % 16:
                    continue
                d_ff = max(step, int(d * d_ff_ratio) // step * step)
                n = _stack_count(arch, d, d_ff, nl)
                if n > target_stack * 1.5:
                    break
                rel = abs(n - target_stack) / max(1, target_stack)
                key = (rel, -min(n, target_stack))
                if best is None or key < best[0]:
                    best = (key, n, rel, d, d_ff, nl)
    _, n, rel, d, d_ff, nl = best
    cfg = LinearConfig(arch=arch, vocab_size=vocab_size, d_model=d,
                       n_layers=nl, d_ff=d_ff, max_len=max_len)
    return cfg, n, rel


def _make(arch: str, cfg: LinearConfig) -> _LMBase:
    base = _LMBase(cfg)
    for _ in range(cfg.n_layers):
        base.blocks.append(BLOCKS[arch](cfg))
    base._finish_init()
    return base


_ARCH_LM.update({
    "linear_attn": lambda cfg: _make("linear_attn", cfg),
    "rwkv": lambda cfg: _make("rwkv", cfg),
    "mamba": lambda cfg: _make("mamba", cfg),
    "hyena": lambda cfg: _make("hyena", cfg),
    "h3": lambda cfg: _make("h3", cfg),
})


class LinearAttentionLM(_LMBase):
    def __init__(self, cfg: LinearConfig):
        cfg.arch = "linear_attn"
        super().__init__(cfg)
        for _ in range(cfg.n_layers):
            self.blocks.append(LinearAttentionBlock(cfg))
        self._finish_init()


class RWKVLM(_LMBase):
    def __init__(self, cfg: LinearConfig):
        cfg.arch = "rwkv"
        super().__init__(cfg)
        for _ in range(cfg.n_layers):
            self.blocks.append(RWKVBlock(cfg))
        self._finish_init()


class MambaLM(_LMBase):
    def __init__(self, cfg: LinearConfig):
        cfg.arch = "mamba"
        super().__init__(cfg)
        for _ in range(cfg.n_layers):
            self.blocks.append(MambaBlock(cfg))
        self._finish_init()


class HyenaLM(_LMBase):
    def __init__(self, cfg: LinearConfig):
        cfg.arch = "hyena"
        super().__init__(cfg)
        for _ in range(cfg.n_layers):
            self.blocks.append(HyenaBlock(cfg))
        self._finish_init()


class H3LM(_LMBase):
    def __init__(self, cfg: LinearConfig):
        cfg.arch = "h3"
        super().__init__(cfg)
        for _ in range(cfg.n_layers):
            self.blocks.append(H3Block(cfg))
        self._finish_init()