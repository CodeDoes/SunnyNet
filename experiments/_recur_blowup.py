"""加法注入在 p->1 (长记忆) 下的发散, 对照凸组合. 同一训练循环, 无 grad clip.

T 拉长到 2048, 保留率初值更高 (p=sigmoid(6)=0.9975), 同样的 lr 与优化器。
目的: 把"训练 RWKV 为什么会爆"从调参归因拉回到递推结构本身。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.manual_seed(0)
T, B, D = 1024, 8, 64
STEPS = 120


class Cell(nn.Module):
    def __init__(self, convex: bool):
        super().__init__()
        self.convex = convex
        self.embed = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, 1, bias=False)
        self.w = nn.Parameter(torch.full((D,), 6.0))   # p=sigmoid(6)=0.9975
        self.kv = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            for p in (self.embed, self.head, self.kv):
                nn.init.normal_(p.weight, std=0.5)

    def forward(self, x):
        h = torch.tanh(self.embed(x))
        p = torch.sigmoid(self.w).view(1, -1)
        g = torch.tanh(self.kv(h))
        s = torch.zeros(x.shape[0], D)
        states = []
        for t in range(x.shape[1]):
            if self.convex:
                s = p * s + (1.0 - p) * g[:, t]
            else:
                s = p * s + g[:, t]
            states.append(s)
        return self.head(torch.stack(states, dim=1)).squeeze(-1)


def run(convex: bool):
    tag = "凸组合" if convex else "加法注入"
    x = torch.randn(B, T, D) * 0.5
    y = torch.randn(B, T) * 0.1
    m = Cell(convex)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(STEPS):
        loss = ((m(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e18))
        if not torch.isfinite(loss) or gn > 1e6:
            print(f"  [{tag}] step {step+1} 爆炸: loss={float(loss):.3e} "
                  f"grad_norm={gn:.3e}")
            return
        opt.step()
    with torch.no_grad():
        h = torch.tanh(m.embed(x))
        g = torch.tanh(m.kv(h))
        p = torch.sigmoid(m.w).view(1, -1)
        s = torch.zeros(B, D)
        mx = 0.0
        for t in range(T):
            s = p * s + ((1.0 - p) * g[:, t] if convex else g[:, t])
            mx = max(mx, float(s.abs().max()))
    print(f"  [{tag}] {STEPS} 步未爆: loss={float(loss):.4f} grad_norm={gn:.3f} "
          f"max|s|={mx:.2f} p_learned={torch.sigmoid(m.w).mean():.5f}")


def main():
    print(f"T={T}, p 初值 0.9975, lr=3e-3, 无 grad clip:")
    run(convex=False)
    run(convex=True)


if __name__ == "__main__":
    main()