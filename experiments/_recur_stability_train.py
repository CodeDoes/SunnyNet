"""两种递推结构在同一训练循环下的梯度/状态行为.

加法注入 (类 RWKV-4/5/6 wkv): s_t = p_t*s_{t-1} + g_t
凸组合   (SunnyNet / Mamba-2):  s_t = p_t*s_{t-1} + (1-p_t)*g_t

两者都用可学保留率、有界输入、同一 lr、同一优化器, 看状态范数与梯度范数随步数。
目的是把"训练 RWKV 为什么会爆"定位到结构, 而不是归因于学习率调参。
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)
T, B, D = 512, 32, 64
STEPS = 300


class Cell(nn.Module):
    def __init__(self, convex: bool):
        super().__init__()
        self.convex = convex
        self.embed = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, 1, bias=False)
        self.w = nn.Parameter(torch.zeros(D))          # 保留率 logit, 可学
        self.kv = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.w.fill_(4.0)                          # 初始 p=sigmoid(4)=0.982
            for p in (self.embed, self.head, self.kv):
                nn.init.normal_(p.weight, std=0.5)

    def recurrence(self, g, p):
        s = torch.zeros(g.shape[0], g.shape[-1], device=g.device)
        states = []
        for t in range(g.shape[1]):
            if self.convex:
                s = p * s + (1.0 - p) * g[:, t]
            else:
                s = p * s + g[:, t]
            states.append(s)
        return torch.stack(states, dim=1)

    def forward(self, x):
        h = torch.tanh(self.embed(x))
        p = torch.sigmoid(self.w).view(1, -1)           # 可学保留率, 无上界参数化
        g = torch.tanh(self.kv(h))
        s = self.recurrence(g, p)
        return self.head(s).squeeze(-1)


def run(convex: bool):
    x = torch.randn(B, T, D) * 0.5
    y = torch.randn(B, T) * 0.1
    m = Cell(convex)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)    # 故意不设 grad clip
    blown = False
    for step in range(STEPS):
        loss = ((m(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e18))
        if not torch.isfinite(loss) or gn > 1e6:
            print(f"  [{'凸组合' if convex else '加法注入'}] step {step+1} 爆炸: "
                  f"loss={float(loss):.3e} grad_norm={gn:.3e}")
            blown = True
            break
        opt.step()
    if not blown:
        with torch.no_grad():
            h = torch.tanh(m.embed(x))
            g = torch.tanh(m.kv(h))
            p = torch.sigmoid(m.w).view(1, 1, -1)
            s = m.recurrence(g, p)
        print(f"  [{'凸组合' if convex else '加法注入'}] {STEPS} 步未爆: "
              f"loss={float(loss):.4f} grad_norm={gn:.3f} max|s|={s.abs().max():.3f} "
              f"p_learned={torch.sigmoid(m.w).mean():.4f}")


def main():
    print("同一训练循环 (无 grad clip, lr=3e-3, 可学保留率):")
    run(convex=False)
    run(convex=True)


if __name__ == "__main__":
    main()