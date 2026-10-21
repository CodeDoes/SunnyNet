"""对照基线: 标准 Transformer 语言模型.

仅用于公平对照 (同参数量 / 同数据 / 同硬件 / 同 token 预算), 不属于 SunnyNet 架构本体.
手写实现以便精确控制参数量与结构, 不调用 nn.TransformerEncoder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BaselineConfig", "TransformerLM", "TransformerState", "solve_baseline_for_params"]


@dataclass
class BaselineConfig:
    vocab_size: int = 12000
    d_model: int = 320
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 640
    max_len: int = 2048
    tie_embeddings: bool = True


@dataclass
class TransformerState:
    """KV cache: 逐层 (k, v) 均为 (B,H,T_past,d_head), 首步前为 None."""

    kv: list
    pos: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def step(self, x_t, cache):
        """单步: x_t (B,C), cache=(k,v) 形如 (B,H,T_past,d_head); 返回 (y, cache_new)."""
        B, C = x_t.shape
        q, k, v = self.qkv(x_t).split(C, dim=-1)
        q = q.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, 1, C)
        return self.proj(y)[:, 0], (k, v)


class Block(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        h = F.gelu(self.fc1(self.ln2(x)))
        return x + self.fc2(h)

    def step(self, x_t, cache):
        y, c = self.attn.step(self.ln1(x_t), cache)
        x_t = x_t + y
        return x_t + self.fc2(F.gelu(self.fc1(self.ln2(x_t)))), c


class TransformerLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        mask = torch.tril(torch.ones(cfg.max_len, cfg.max_len)).view(1, 1, cfg.max_len, cfg.max_len)
        self.register_buffer("mask", mask, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            n -= self.pos.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.head.weight.numel()
        return n

    def forward(self, idx, state=None, return_state=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x, self.mask)
        logits = self.head(self.ln_f(x))
        if return_state:
            return logits, None
        return logits

    @torch.no_grad()
    def forward_step(self, idx_t, state):
        """单步自回归 (KV cache): idx_t (B,) -> (logits (B,V), state)."""
        if state is None:
            state = TransformerState(kv=[None] * len(self.blocks), pos=0)
        x = self.embed(idx_t) + self.pos(torch.tensor(state.pos, device=idx_t.device))
        new_kv = []
        for blk, c in zip(self.blocks, state.kv):
            x, cn = blk.step(x, c)
            new_kv.append(cn)
        logits = self.head(self.ln_f(x))
        return logits, TransformerState(kv=new_kv, pos=state.pos + 1)


def stack_param_count(d_model: int, d_ff: int, n_layers: int) -> int:
    """Transformer 计算栈参数量, 不含 token/位置嵌入与 LM head.

    SunnyNet 无位置嵌入, 只有排除位置嵌入两边口径才一致.
    """
    per_layer = (d_model * 3 * d_model + 3 * d_model
                 + d_model * d_model + d_model
                 + 2 * d_model
                 + d_model * d_ff + d_ff
                 + d_ff * d_model + d_model
                 + 2 * d_model)
    return n_layers * per_layer + 2 * d_model


def solve_baseline_for_params(target_non_embed: int, vocab_size: int,
                              n_layers: int = 6, n_heads: int = 8,
                              max_len: int = 2048):
    """枚举 (d_model, d_ff) 使计算栈参数量最接近目标. 用解析公式, 不实例化模型.

    返回 (cfg, 实际栈参数量, 相对偏差).
    """
    best = None
    for d_model in range(64, 3072, 8):
        if d_model % n_heads:
            continue
        base = (n_layers * (d_model * 3 * d_model + 3 * d_model
                            + d_model * d_model + d_model + 2 * d_model
                            + d_model * d_model + d_model + 2 * d_model)
                + 2 * d_model)
        if base > target_non_embed * 1.05:
            break
        for d_ff in range(d_model, 6 * d_model + 1, 16):
            n = stack_param_count(d_model, d_ff, n_layers)
            rel = abs(n - target_non_embed) / max(1, target_non_embed)
            key = (rel, -min(n, target_non_embed))
            if best is None or key < best[0]:
                best = (key, n, rel, BaselineConfig(
                    vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers, d_ff=d_ff, max_len=max_len))
    _, n, rel, cfg = best
    return cfg, n, rel