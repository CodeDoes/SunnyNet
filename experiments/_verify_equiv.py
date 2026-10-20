"""验证三项优化的数学等价性 (fp64):
1. W_s(circ(x)) == x @ W_eff^T,  W_eff = t0*roll(W,-1,1) + t1*W + t2*roll(W,1,1)
2. hist 全零时 _apply_local(raw,hist) == raw + dw(raw)
3. softwedge(z,a) 与 z*(1+0.5a/(1+|z|)) 在 CenteredScaleNorm 之后等价
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sunnynet.layers import DepthwiseCausal, StateCirculant  # noqa: E402
from sunnynet.ops import CenteredScaleNorm, softwedge  # noqa: E402

torch.manual_seed(0)
D, B, T = 128, 4, 64
dev, dt = "cpu", torch.float64

# --- 1. circ + Linear 折叠 ---
circ = StateCirculant(D).to(dt)
W = torch.randn(D, D, dtype=dt) * 0.1
x = torch.randn(B, T, D, dtype=dt)
ref = F.linear(circ(x), W)
t = circ.t.to(dt)
W_eff = t[0] * torch.roll(W, -1, dims=1) + t[1] * W + t[2] * torch.roll(W, 1, dims=1)
fold = F.linear(x, W_eff)
print(f"1. circ+Linear 折叠  maxdiff = {(ref - fold).abs().max().item():.3e}  "
      f"scale={ref.abs().max().item():.4f}")

# --- 2. hist 全零时 _apply_local 免 cat ---
dw = DepthwiseCausal(D, taps=4).to(dt)
raw = torch.randn(B, T, D, dtype=dt)
hist0 = torch.zeros(B, 3, D, dtype=dt)
k = 3
rf = torch.cat([hist0, raw], dim=1)
ref2 = (rf + dw(rf))[:, k:]
fast2 = raw + dw(raw)
print(f"2. _apply_local 免 cat  maxdiff = {(ref2 - fast2).abs().max().item():.3e}  "
      f"scale={ref2.abs().max().item():.4f}")

# 非零 hist 时应走原路 (验证两路确实不同)
histn = torch.randn(B, 3, D, dtype=dt)
rfn = torch.cat([histn, raw], dim=1)
ref2n = (rfn + dw(rfn))[:, k:]
print(f"   非零 hist 与免 cat 差异 = {(ref2n - fast2).abs().max().item():.3e} "
      f"(应显著 >0, 证明必须保留原路)")

# --- 3. softwedge 常数项在下游 norm 后等价 ---
norm = CenteredScaleNorm(D).to(dt)
z = torch.randn(B, T, D, dtype=dt) * 3
a = torch.tensor(0.5, dtype=dt)
ref3 = norm(softwedge(z, a))
centered = z * (1.0 + 0.5 * a / (1.0 + z.abs()))
fast3 = norm(centered)
print(f"3. softwedge 常数折叠 maxdiff = {(ref3 - fast3).abs().max().item():.3e}  "
      f"scale={ref3.abs().max().item():.4f}")

ok = True
for name, d, sc in (("circ折叠", (ref - fold).abs().max().item(), ref.abs().max().item()),
                    ("免cat", (ref2 - fast2).abs().max().item(), ref2.abs().max().item()),
                    ("softwedge", (ref3 - fast3).abs().max().item(), ref3.abs().max().item())):
    rel = d / max(sc, 1e-12)
    good = rel < 1e-12
    ok &= good
    print(f"   {name}: 相对误差 {rel:.2e}  {'通过' if good else '不合格'}")

print("VERDICT:", "全部等价" if ok else "存在不等价项")