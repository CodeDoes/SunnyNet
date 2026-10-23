"""状态有界性的直接对照: SunnyNet 凸组合 vs RWKV-7 rank-1 累加.

两段递推在同样的"衰减可学、输入有界"前提下跑长序列, 看状态范数随时间的行为。
不训练, 只推进状态; 目的是把"哪种更新结构本身不会累加发散"变成可复现读数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

rng = np.random.default_rng(0)
T = 4096
D = 256


def sunny_state_norm(u, p, s0):
    """s_t = p_t*s_{t-1} + (1-p_t)*u_t, p in (0,1). 凸组合."""
    s = s0.copy()
    out = np.empty(T)
    for t in range(T):
        s = p[t] * s + (1.0 - p[t]) * u[t]
        out[t] = np.abs(s).max()
    return out


def rwkv7_state_norm(w, kk, a, v, k, S0, nh):
    """S_t = S*w - outer(S kk, kk*a) + outer(v, k). 逐 head 矩阵, rank-1 校正."""
    S = S0.copy()
    out = np.empty(T)
    for t in range(T):
        for h in range(nh):
            Sh = S[h]
            proj = Sh @ kk[t, h]
            S[h] = (Sh * w[t, h][None, :]
                    - np.outer(proj, kk[t, h] * a[t, h])
                    + np.outer(v[t, h], k[t, h]))
        out[t] = np.abs(S).max()
    return out


def main():
    nh, hs = 8, D // 8
    # 输入有界 (组内竞争门输出乘以 w<1, 这里取均匀采样模拟有界候选状态)
    u = rng.uniform(-1.0, 1.0, size=(T, D)).astype(np.float32)
    p = rng.uniform(0.99, 0.9999, size=(T, D)).astype(np.float32)   # 长记忆: 保留率高

    s0 = np.zeros(D, dtype=np.float32)
    sn = sunny_state_norm(u, p, s0)
    max_u = np.abs(u).max()
    print(f"[SunnyNet] max|u| = {max_u:.4f}, max|s_T| = {sn.max():.4f}, "
          f"界 max(|s0|, max|u|) = {max(max_u, 0.0):.4f}  -> "
          f"{'有界(<=max|u|)' if sn.max() <= max_u + 1e-4 else '越界'}")

    # RWKV-7: w 衰减同样接近 1 (长记忆), kk L2 归一化, a in (0,1), v/k 有界
    w = rng.uniform(0.99, 0.9999, size=(T, nh, hs)).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, size=(T, nh, hs)).astype(np.float32)
    k = rng.uniform(-1.0, 1.0, size=(T, nh, hs)).astype(np.float32)
    kk = rng.normal(size=(T, nh, hs)).astype(np.float32)
    kk /= np.maximum(np.linalg.norm(kk, axis=2, keepdims=True), 1e-12)
    a = rng.uniform(0.0, 1.0, size=(T, nh, hs)).astype(np.float32)
    S0 = np.zeros((nh, hs, hs), dtype=np.float32)
    rn = rwkv7_state_norm(w, kk, a, v, k, S0, nh)
    print(f"[RWKV-7]   max|S_0| = 0.0, max|S_T| = {rn.max():.4f}, "
          f"末值 max|S| = {rn[-1]:.4f}")
    # 增长斜率: 前 1/4 与后 1/4 的均值比
    q = T // 4
    print(f"[RWKV-7]   前 1/4 均值 {rn[:q].mean():.4f} -> 后 1/4 均值 {rn[-q:].mean():.4f} "
          f"(比 {rn[-q:].mean() / max(rn[:q].mean(), 1e-9):.2f}x)")
    sq = sn[:q].mean(), sn[-q:].mean()
    print(f"[SunnyNet] 前 1/4 均值 {sq[0]:.4f} -> 后 1/4 均值 {sq[1]:.4f} "
          f"(比 {sq[1] / max(sq[0], 1e-9):.2f}x)")


if __name__ == "__main__":
    main()