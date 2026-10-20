"""窄残差流配置 (dim=80/88/96) 的架构不变量核验.

(fp64) 单步 vs 整段、分段续算 vs 整段、分块不变性、因果性.
用法: python -X utf8 experiments/_invariants_narrow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

V = 8192
CONFIGS = [
    ("dim80", 80, 160, 244),
    ("dim88", 88, 176, 196),
    ("dim96", 96, 192, 152),
]

for name, dim, inner, rank in CONFIGS:
    torch.manual_seed(1234)
    cfg = SunnyConfig(vocab_size=V, dim=dim, inner=inner, rank=rank,
                      layers=4, h_max=2048.0)
    cfg.scan_chunk = 64
    m = SunnyNet(cfg).double().eval()
    x = torch.randint(0, V, (2, 40))

    with torch.no_grad():
        full, s_full = m(x, return_state=True)

        # 单步 vs 整段
        st = None
        outs = []
        for t in range(x.shape[1]):
            if st is None:
                st = m.init_state(x.shape[0], dtype=torch.float64)
            o, st = m.forward_step(x[:, t], st)
            outs.append(o)
        stepped = torch.stack(outs, dim=1)
        d_step = float((full - stepped).abs().max())

        # 分段续算
        o1, s1 = m(x[:, :17], return_state=True)
        o2, _ = m(x[:, 17:], s1, return_state=True)
        seg = torch.cat([o1, o2], dim=1)
        d_seg = float((full - seg).abs().max())

        # 分块不变性
        cfg2 = SunnyConfig(vocab_size=V, dim=dim, inner=inner, rank=rank,
                           layers=4, h_max=2048.0)
        ds = []
        for ch in (1, 7, 64, 128):
            cfg2.scan_chunk = ch
            m2 = SunnyNet(cfg2).double().eval()
            m2.load_state_dict(m.state_dict())
            with torch.no_grad():
                o2_, _ = m2(x, return_state=True)
            ds.append(float((full - o2_).abs().max()))
        d_chunk = max(ds)

        # 因果性: 改末位 token 不应影响前缀输出
        xb = x.clone()
        xb[:, -1] = (xb[:, -1] + 1) % V
        ob, _ = m(xb, return_state=True)
        d_causal = float((full[:, :-1] - ob[:, :-1]).abs().max())

    print(f"{name} stack={SunnyNet(cfg).num_params(non_embedding=True):,} "
          f"step={d_step:.2e} seg={d_seg:.2e} chunk={d_chunk:.2e} causal={d_causal:.2e}")