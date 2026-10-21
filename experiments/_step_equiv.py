"""校验 O(N) 同行单步解码接口与整段前向数值一致 (逐 token 拼回 vs 一次前向)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from sunnynet.linear_baselines import build_linear_baseline, _make

torch.manual_seed(0)
V, T, B = 512, 64, 2
for arch in ["linear_attn", "rwkv", "mamba", "hyena", "h3"]:
    cfg, n, rel = build_linear_baseline(arch, 40000, V, max_len=T + 8, n_layers=2)
    m = _make(arch, cfg).eval()
    idx = torch.randint(0, V, (B, T))
    with torch.no_grad():
        full = m(idx)
        st = m.init_state(B)
        outs = []
        for t in range(T):
            lg, st = m.forward_step(idx[:, t], st)
            outs.append(lg)
        stepwise = torch.stack(outs, dim=1)
    d = (full - stepwise).abs().max().item()
    rel_scale = full.abs().max().item()
    flag = "OK" if d < 1e-4 * max(1.0, rel_scale) else "FAIL"
    print(f"{arch:12s} maxdiff={d:.3e}  scale={rel_scale:.2f}  {flag}")