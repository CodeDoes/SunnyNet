"""通用递推内核的跨架构等价验证.

对每家架构手工取出其递推形式 (a_t, b_t/u_t), 用 runtime 内核跑扫描,
与各家自身的 PyTorch 递推逐元素比对。这是"统一 IR 成立"的直接证据。

口径: fp32; 比对 max|diff|; 阈值 1e-5 (内核与 torch 均 fp32, 递推链无重排).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sunnynet.runtime import scan, scan_bwd, FMA, GATED  # noqa: E402
from sunnynet.ops import gated_scan  # noqa: E402


def ref_recur(a, b, s0, gated):
    """torch 逐步递推参照 (fp64 计, 再回 fp32 比对)."""
    A, b_ = a.double(), b.double()
    s = s0.double()
    outs = []
    for t in range(A.shape[1]):
        at = A[:, t]
        bt = (1.0 - at) * b_[:, t] if gated else b_[:, t]
        s = at * s + bt
        outs.append(s)
    return torch.stack(outs, 1).float(), s.float()


def main():
    torch.manual_seed(0)
    B, T, N = 4, 256, 128
    rows = []

    # 1) FMA 形式: a 任意, b 任意
    a = torch.rand(B, T, N) * 0.9 + 0.05
    b = torch.randn(B, T, N)
    s0 = torch.zeros(B, N)
    o_ref, e_ref = ref_recur(a, b, s0, False)
    o_r, e_r = scan(a, b, s0, gated=False)
    rows.append(("FMA 任意系数", (e_r - e_ref).abs().max().item()))

    # 2) GATED 形式: s = a*s + (1-a)*u, a in (0,1]
    u = torch.randn(B, T, N)
    o_ref, e_ref = ref_recur(a, u, s0, True)
    o_r, e_r = scan(a, u, s0, gated=True)
    rows.append(("GATED (1-a)*u", (e_r - e_ref).abs().max().item()))

    # 3) 与 ops.gated_scan 对齐: p 近 1 的极端值 (SunnyNet 实测 p>=0.5)
    p = torch.rand(B, T, N) * 0.5 + 0.5
    g = torch.randn(B, T, N)
    o_ref, e_ref = gated_scan(p, g, s0)
    # gated_scan 解 s = p*s + g; 映射到 FMA 形式 a=p, b=g
    o_r, e_r = scan(p, g, s0, gated=False)
    rows.append(("对齐 ops.gated_scan", max((o_r - o_ref).abs().max().item(),
                                            (e_r - e_ref).abs().max().item())))

    # 4) logspace_scan 语义: s = p*s + (1-p)*nu
    nu = torch.randn(B, T, N)
    o_ref = torch.empty_like(nu)
    s = s0.clone()
    for t in range(T):
        s = p[:, t] * s + (1.0 - p[:, t]) * nu[:, t]
        o_ref[:, t] = s
    o_r, e_r = scan(p, nu, s0, gated=True)
    rows.append(("logspace_scan 语义", (o_r - o_ref).abs().max().item()))

    # 5) 反向: 与 autograd 对齐 (FMA 形式)
    a2 = (torch.rand(B, T, N) * 0.9 + 0.05).requires_grad_(True)
    b2 = torch.randn(B, T, N).requires_grad_(True)
    s02 = torch.zeros(B, N, requires_grad=True)
    s = s02.clone()
    store = []
    for t in range(T):
        s = a2[:, t] * s + b2[:, t]
        store.append(s)
    out = torch.stack(store, 1)
    w = torch.randn_like(out)
    (out * w).sum().backward()
    with torch.no_grad():
        o_c = out.detach()
        da, db, ds0 = scan_bwd(a2.detach(), b2.detach(), s02.detach(), o_c, w, gated=False)
    rows.append(("反向 dа", (da - a2.grad).abs().max().item()))
    rows.append(("反向 db", (db - b2.grad).abs().max().item()))
    rows.append(("反向 ds0", (ds0 - s02.grad).abs().max().item()))

    # 6) GATED 反向: 对 u 求导应为 ds*(1-a)
    u2 = torch.randn(B, T, N).requires_grad_(True)
    a3 = (torch.rand(B, T, N) * 0.9 + 0.05)
    s = torch.zeros(B, N)
    store = []
    for t in range(T):
        s = a3[:, t] * s + (1.0 - a3[:, t]) * u2[:, t]
        store.append(s)
    out = torch.stack(store, 1)
    w = torch.randn_like(out)
    (out * w).sum().backward()
    with torch.no_grad():
        da3, du3, _ = scan_bwd(a3, u2.detach(), torch.zeros(B, N),
                               out.detach(), w, gated=True)
    rows.append(("GATED 反向 du", (du3 - u2.grad).abs().max().item()))

    # 7) 单步 vs 整段一致 (FMA 形式, 注意 torch 切片非连续需 contiguous)
    from sunnynet.runtime import step
    a4 = torch.rand(B, T, N) * 0.9 + 0.05
    b4 = torch.randn(B, T, N)
    o_ref4, _ = ref_recur(a4, b4, s0, False)
    o_scan4, _ = scan(a4, b4, s0, gated=False)
    s = s0.clone()
    steps = []
    for t in range(T):
        s = step(a4[:, t].contiguous(), b4[:, t].contiguous(), s, gated=False)
        steps.append(s.clone())
    o_st = torch.stack(steps, 1)
    rows.append(("单步 vs 整段", max((o_st - o_ref4).abs().max().item(),
                                    (o_scan4 - o_ref4).abs().max().item())))

    print(f"{'用例':<26} {'max|diff|':>12}  判定")
    print("-" * 52)
    ok = True
    for name, d in rows:
        good = d < 1e-5
        ok = ok and good
        print(f"{name:<26} {d:>12.3e}  {'通过' if good else '不通过'}")
    print("-" * 52)
    print("全部通过" if ok else "存在不通过项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())