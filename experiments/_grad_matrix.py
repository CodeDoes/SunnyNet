"""测量 24 层 / 4 层梯度范数随 rho0 / decay_beta / residual 的矩阵.

用途: 在 CenteredScaleNorm 真实归一化 (s_min 下调) 之后, 重新标定深度稳定性
回归防线的阈值与依据。每格独立随机种子, 只测嵌入梯度范数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sunnynet.model import SunnyConfig, SunnyNet


def grad_norm(L, rho0, beta, residual, s_min, seed=7):
    torch.manual_seed(seed)
    cfg = SunnyConfig(vocab_size=256, dim=128, inner=192, rank=64,
                      layers=L, rho0=rho0, decay_beta=beta,
                      ablation={"residual": residual})
    m = SunnyNet(cfg)
    for b in m.blocks:
        b.norm_in.s_min = s_min
        b.norm_mix.s_min = s_min
    if hasattr(m, "norm_out"):
        m.norm_out.s_min = s_min
    idx = torch.randint(0, 256, (2, 128))
    loss = torch.nn.functional.cross_entropy(
        m(idx).reshape(-1, 256), idx.reshape(-1))
    m.zero_grad(set_to_none=True)
    loss.backward()
    g = m.embed.weight.grad
    return g.norm().item() if g is not None else 0.0


if __name__ == "__main__":
    out = []
    for s_min in (0.02, 0.005, 0.05, 0.25):
        for residual in (False, True):
            for rho0, beta in ((0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.5, 0.0),
                               (0.7, 0.0), (1.0, 0.0), (1.0, 0.15), (2.0, 0.0)):
                g4 = grad_norm(4, rho0, beta, residual, s_min)
                g24 = grad_norm(24, rho0, beta, residual, s_min)
                rec = dict(s_min=s_min, residual=residual, rho0=rho0, beta=beta,
                           g4=g4, g24=g24, ratio=g24 / max(g4, 1e-30))
                out.append(rec)
                print(json.dumps(rec), flush=True)
    print("RESULT", json.dumps(out))