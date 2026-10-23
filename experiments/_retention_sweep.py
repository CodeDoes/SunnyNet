"""三种递推结构在"保留率 p -> 1"时的稳态行为对照.

加法注入 (RWKV-4 wkv):     s_t = p*s_{t-1} + g_t     稳态 ~ g/(1-p) -> 无界
凸组合   (SunnyNet/Mamba2): s_t = p*s_{t-1}+(1-p)*g_t 稳态 ~ g        -> 有界
rank-1 校正 (RWKV-7):       S_t = S*w - outer(S kk, kk a) + outer(v,k)

读数是同一个有界输入 g 下, 稳态幅度随 p 的变化。这是"训练会不会爆炸"的结构根因。
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
D = 64
T = 4000


def steady_additive(p, g):
    s = np.zeros_like(g[0])
    for t in range(T):
        s = p * s + g[t]
    return np.abs(s).max()


def steady_convex(p, g):
    s = np.zeros_like(g[0])
    for t in range(T):
        s = p * s + (1.0 - p) * g[t]
    return np.abs(s).max()


def main():
    g = rng.uniform(-1.0, 1.0, size=(T, D)).astype(np.float32)
    gmax = np.abs(g).max()
    print(f"输入 max|g| = {gmax:.4f}  (每步同分布)\n")
    print(f"{'保留率 p':>10} {'加法稳态':>12} {'/输入':>8} {'凸组合稳态':>12} {'/输入':>8}")
    for p in (0.9, 0.99, 0.999, 0.9999, 1.0 - 1e-6):
        a = steady_additive(p, g)
        c = steady_convex(p, g)
        print(f"{p:>10.6f} {a:>12.3f} {a/gmax:>8.2f} {c:>12.3f} {c/gmax:>8.2f}")
    print("\n加法稳态 ~ g/(1-p), p->1 时线性发散; 凸组合恒 <= max|g|。")


if __name__ == "__main__":
    main()