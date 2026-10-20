"""质量 A/B: 手写 CSN + 逐块串行扫描 (旧) vs 融合 LN + 两级并行扫描 (新).

协议: 600 步, T=256, batch=8, lr=3e-3, warmup=60, AdamW(0.9,0.95), 每格独立子进程.
用法: python -X utf8 experiments/_quality_ab.py <impl> <seed> [steps] [T] [B]
     impl: old | new
输出: RESULT {json}
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def old_csn(self, x):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
    return self.gamma * (x - mu + self.center) * inv


def old_scan(nu, p, s0, chunk=64, p_min_hint=None):
    import math as _m
    g = (1.0 - p) * nu
    p, g = torch.broadcast_tensors(p, g)
    T = p.shape[-2]
    lp = torch.log(p)
    pm = p.detach().amin().item() if p_min_hint is None else p_min_hint
    if pm <= 0 or pm >= 1:
        ch = min(chunk, 64)
    else:
        ch = max(1, min(chunk, int(64 * _m.log(2.0) / (-_m.log(pm)))))
    s = s0
    outs = []
    for st in range(0, T, ch):
        en = min(st + ch, T)
        P = torch.cumsum(lp[..., st:en, :], dim=-2)
        A = torch.cumsum(g[..., st:en, :] * torch.exp(-P), dim=-2)
        sc = torch.exp(P) * (s.unsqueeze(-2) + A)
        outs.append(sc)
        s = sc[..., -1, :]
    return torch.cat(outs, dim=-2), s


def main():
    impl = sys.argv[1]
    seed = int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    B = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    torch.set_num_threads(8)
    vocab = 8192
    torch.manual_seed(seed)

    if impl == "old":
        import sunnynet.layers as L
        import sunnynet.ops as O
        from sunnynet.ops import CenteredScaleNorm
        CenteredScaleNorm.forward = old_csn
        O.logspace_scan = old_scan
        L.logspace_scan = old_scan

    m = SunnyNet(SunnyConfig.small(vocab)).train()

    # 固定语料 (同 seed 下两 impl 看到相同数据)
    g = torch.Generator().manual_seed(999)
    data = torch.randint(0, vocab, (B, T * 40), generator=g)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    warm = max(1, steps // 10)

    def lr_at(it):
        if it < warm:
            return 3e-3 * (it + 1) / warm
        p = (it - warm) / max(1, steps - warm)
        return 3e-4 + 0.5 * (3e-3 - 3e-4) * (1 + math.cos(math.pi * p))

    losses = []
    t0 = time.perf_counter()
    for it in range(steps):
        i = (it * T) % (data.shape[1] - T - 1)
        xb = data[:, i:i + T]
        yb = data[:, i + 1:i + T + 1]
        lg = m(xb)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for grp in opt.param_groups:
            grp["lr"] = lr_at(it)
        opt.step()
        losses.append(loss.item())
    wall = time.perf_counter() - t0

    # 持出验证: 数据尾部
    with torch.no_grad():
        m.eval()
        vs = []
        nv = 8
        for k in range(nv):
            i = data.shape[1] - (k + 1) * T - 1
            if i < 0:
                break
            xb = data[:, i:i + T]
            yb = data[:, i + 1:i + T + 1]
            lg = m(xb)
            vs.append(torch.nn.functional.cross_entropy(
                lg.reshape(-1, vocab), yb.reshape(-1)).item())
        val = statistics.mean(vs) if vs else float("nan")

    print("RESULT " + json.dumps({
        "impl": impl, "seed": seed, "steps": steps, "T": T, "B": B,
        "train_loss_last": round(statistics.mean(losses[-20:]), 4),
        "val_loss": round(val, 5), "ppl": round(math.exp(val), 2),
        "wall_s": round(wall, 1),
        "ms_per_step": round(wall / steps * 1000, 1),
    }))


if __name__ == "__main__":
    main()