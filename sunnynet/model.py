"""SunnyNet 模型主体 (原创通用序列神经网络).

结构: 嵌入 -> N 个 SunnyBlock (滚动状态逐层接力, 无残差相加) -> 终态归一 -> 绑定 LM Head.

跨层信息传递: 第 l 块的末态 s_l 直接作为第 l+1 块的初始状态, 中间不经过任何
加法旁路. 尺度由 CenteredScaleNorm 与谱锚定初始化共同锁定, 深度靠状态自身承载.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .init import spectral_anchored_init
from .layers import SunnyBlock, SunnyState
from .ops import CenteredScaleNorm

__all__ = ["SunnyConfig", "SunnyNet", "SunnyState"]


@dataclass
class SunnyConfig:
    vocab_size: int = 12000
    dim: int = 256
    inner: int = 384
    rank: int = 128
    taps: int = 4
    layers: int = 6
    scan_chunk: int = 64
    rho0: float = 1.0
    decay_beta: float = 0.0
    h_min: float = 2.0
    h_max: float = 8192.0
    tie_embeddings: bool = True
    ablation: dict = field(default_factory=dict)

    @staticmethod
    def small(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=128, inner=192, rank=64,
                           layers=4, h_max=2048.0)

    @staticmethod
    def proto20m(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=512, inner=896, rank=320,
                           layers=10, h_max=8192.0)

    @staticmethod
    def scale05b(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=1536, inner=2560, rank=896,
                           layers=16, h_max=32768.0)


class SunnyNet(nn.Module):
    def __init__(self, cfg: SunnyConfig):
        super().__init__()
        self.cfg = cfg
        ab = cfg.ablation
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([
            SunnyBlock(cfg.dim, cfg.inner, rank=cfg.rank, taps=cfg.taps,
                       layer_index=i, rho0=cfg.rho0, decay_beta=cfg.decay_beta,
                       h_min=cfg.h_min, h_max=cfg.h_max,
                       scan_chunk=cfg.scan_chunk, ablation=ab)
            for i in range(cfg.layers)
        ])
        self.norm_out = CenteredScaleNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.head.weight.numel()
        return n

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> SunnyState:
        k = max(0, self.cfg.taps - 1)
        return SunnyState(
            s=torch.zeros(batch, self.cfg.layers, self.cfg.dim, device=device, dtype=dtype),
            hist=torch.zeros(batch, self.cfg.layers, k, self.cfg.dim,
                             device=device, dtype=dtype))

    def forward(self, idx: torch.Tensor, state: SunnyState | None = None,
                return_state: bool = False):
        """idx (B,T) -> logits (B,T,V); state 为 SunnyState 或 None."""
        B, T = idx.shape
        x = self.embed(idx)
        k = max(0, self.cfg.taps - 1)
        if state is None:
            s = torch.zeros(B, self.cfg.layers, self.cfg.dim, device=idx.device, dtype=x.dtype)
            hist = torch.zeros(B, self.cfg.layers, k, self.cfg.dim,
                               device=idx.device, dtype=x.dtype)
        else:
            s, hist = state.s, state.hist
        ns = torch.empty_like(s) if return_state else None
        nh = torch.empty_like(hist) if return_state else None
        for i, blk in enumerate(self.blocks):
            x, s_i, h_i = blk(x, s[:, i], hist[:, i])
            if return_state:
                ns[:, i] = s_i
                nh[:, i] = h_i
        logits = self.head(self.norm_out(x))
        if return_state:
            return logits, SunnyState(ns, nh)
        return logits

    @torch.no_grad()
    def forward_step(self, idx_t: torch.Tensor, state: SunnyState):
        """单步自回归: idx_t (B,) -> (logits (B,V), SunnyState)."""
        x = self.embed(idx_t)
        ns = torch.empty_like(state.s)
        nh = torch.empty_like(state.hist)
        for i, blk in enumerate(self.blocks):
            x, s_i, h_i = blk.step(x, state.s[:, i], state.hist[:, i])
            ns[:, i] = s_i
            nh[:, i] = h_i
        logits = self.head(self.norm_out(x))
        return logits, SunnyState(ns, nh)

    def state_dict_for_inference(self, batch: int, device=None, dtype=torch.float32):
        return self.init_state(batch, device, dtype)