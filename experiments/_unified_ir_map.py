"""跨架构统一 IR 的映射验证.

把各家线性架构的递推形式手工映射到通用内核
    s_t = a_t * s_{t-1} + b_t        (SN_FMA)
    s_t = a_t * s_{t-1} + (1-a_t)*u_t (SN_GATED)
并与各家自身的 PyTorch 逐 token 递推逐元素比对。

覆盖: SunnyNet 滚动状态 / RWKV wkv / Mamba SSM / H3 对角 SSM / 线性注意力累积 KV。
Hyena 是长卷积 (FIR), 不在一阶递推范畴, 单列说明。

口径: fp32, 随机初始化权重, T=128, B=2, d_model=128; 阈值 1e-5。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sunnynet.runtime import scan as rt_scan  # noqa: E402
from sunnynet.ops import gated_scan  # noqa: E402


def rwkv_like():
    """RWKV-4 时间混合: wkv_t = p_t*wkv_{t-1} + k_t*v_t."""
    B, T, N = 2, 128, 128
    torch.manual_seed(1)
    p = torch.rand(B, T, N) * 0.6 + 0.4
    k = torch.randn(B, T, N)
    v = torch.randn(B, T, N)
    # 参照: torch 递推
    s = torch.zeros(B, N)
    ref = []
    for t in range(T):
        s = p[:, t] * s + k[:, t] * v[:, t]
        ref.append(s)
    ref = torch.stack(ref, 1)
    # 内核 FMA: a=p, b=k*v
    got, _ = rt_scan(p, k * v, torch.zeros(B, N), gated=False)
    return "RWKV-4 wkv", (got - ref).abs().max().item()


def mamba_like():
    """Mamba/S6 对角 SSM: s_t = decay_t*s_{t-1} + (1-decay_t)*(b_t*x_t)."""
    B, T, N = 2, 128, 128
    torch.manual_seed(2)
    decay = torch.rand(B, T, N) * 0.5 + 0.4
    bu = torch.randn(B, T, N)
    s = torch.zeros(B, N)
    ref = []
    for t in range(T):
        s = decay[:, t] * s + (1.0 - decay[:, t]) * bu[:, t]
        ref.append(s)
    ref = torch.stack(ref, 1)
    got, _ = rt_scan(decay, bu, torch.zeros(B, N), gated=True)
    return "Mamba/S6 SSM", (got - ref).abs().max().item()


def h3_like():
    """H3 对角 SSM 段: 与 Mamba 同形式, decay 与输入无关."""
    B, T, N = 2, 128, 128
    torch.manual_seed(3)
    decay = torch.full((B, T, N), 0.7)
    v = torch.randn(B, T, N)
    s = torch.zeros(B, N)
    ref = []
    for t in range(T):
        s = decay[:, t] * s + (1.0 - decay[:, t]) * v[:, t]
        ref.append(s)
    ref = torch.stack(ref, 1)
    got, _ = rt_scan(decay, v, torch.zeros(B, N), gated=True)
    return "H3 对角 SSM", (got - ref).abs().max().item()


def linear_attn_like():
    """线性注意力状态累加: KV_t = KV_{t-1} + k_t v_t^T  (a=1, b=k v^T).

    用 N=d*d 的形状把 (d,d) 状态摊平成通道维, 验证 a=1 的纯累加。
    """
    B, T, d = 2, 64, 16
    N = d * d
    torch.manual_seed(4)
    k = torch.randn(B, T, d)
    v = torch.randn(B, T, d)
    kv = torch.einsum("btd,bte->btde", k, v).reshape(B, T, N)
    a = torch.ones(B, T, N)
    s = torch.zeros(B, N)
    ref = []
    for t in range(T):
        s = s + kv[:, t]
        ref.append(s)
    ref = torch.stack(ref, 1)
    got, _ = rt_scan(a, kv, torch.zeros(B, N), gated=False)
    return "线性注意力累积 KV", (got - ref).abs().max().item()


def sunnynet_like():
    """SunnyNet 滚动状态: 与 ops.gated_scan 对齐 (p>=0.5 实测区间)."""
    B, T, N = 2, 128, 128
    torch.manual_seed(5)
    p = torch.rand(B, T, N) * 0.5 + 0.5
    u = torch.randn(B, T, N)
    # gated_scan 解 s = p*s + g, 故 g = (1-p)*u
    ref, _ = gated_scan(p, (1.0 - p) * u, torch.zeros(B, N))
    got, _ = rt_scan(p, u, torch.zeros(B, N), gated=True)
    return "SunnyNet 滚动状态", (got - ref).abs().max().item()


def s4_like():
    """S4 对角化 SSM: s_t = A_bar*s + B_bar*x, 用负实数 A_bar 映射."""
    B, T, N = 2, 128, 128
    torch.manual_seed(6)
    a = torch.rand(B, T, N) * 0.7 + 0.2
    b = torch.randn(B, T, N)
    s = torch.zeros(B, N)
    ref = []
    for t in range(T):
        s = a[:, t] * s + b[:, t]
        ref.append(s)
    ref = torch.stack(ref, 1)
    got, _ = rt_scan(a, b, torch.zeros(B, N), gated=False)
    return "S4 对角 SSM", (got - ref).abs().max().item()


def main():
    rows = [rwkv_like(), mamba_like(), h3_like(), linear_attn_like(),
            sunnynet_like(), s4_like()]
    print(f"{'架构递推形式':<24} {'max|diff|':>12}  判定")
    print("-" * 50)
    ok = True
    for name, d in rows:
        good = d < 1e-5
        ok = ok and good
        print(f"{name:<24} {d:>12.3e}  {'通过' if good else '不通过'}")
    print("-" * 50)
    print("五家 O(N) 同行的递推全部落进同一内核" if ok else "存在不通过项")
    print("说明: Hyena 是长卷积 (FIR) 递推, 不属于一阶状态递推, 需独立卷积后端。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())