"""SunnyNet 训练步速度: 单格独立子进程, 多轮重复取中位.

用法: python -X utf8 experiments/_speed_ab.py <impl> <T> <B> <threads> [rounds]
     impl: cur (当前栈) | a (orig gated_scan) | b (no cumprod)
输出: RESULT {json}
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def old_gated_scan(p, g, s0, chunk=64, p_min_hint=None):
    """修正前的逐块串行扫描 (dim=-2 cumsum)."""
    import math
    p, g = torch.broadcast_tensors(p, g)
    T = p.shape[-2]
    lp = torch.log(p)
    if p_min_hint is None:
        p_min = p.detach().amin().item()
    else:
        p_min = p_min_hint
    if p_min <= 0 or p_min >= 1:
        ch = min(chunk, 64)
    else:
        ch = max(1, min(chunk, int(64 * math.log(2.0) / (-math.log(p_min)))))
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


def old_csn(self, x):
    """修正前的手写 CSN."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    inv = torch.rsqrt(var + self.s_min * self.s_min + self.eps)
    return self.gamma * (x - mu + self.center) * inv


def apply_impl(impl):
    import sunnynet.layers as L
    import sunnynet.ops as O
    from sunnynet.ops import CenteredScaleNorm

    if impl == "a":
        # 仅回退扫描 (CSN 用新的融合核)
        O.logspace_scan = lambda nu, p, s0, chunk=64, p_min_hint=None: \
            old_gated_scan(p, (1.0 - p) * nu, s0, chunk, p_min_hint)
        L.logspace_scan = O.logspace_scan
    elif impl == "b":
        # 仅回退 CSN (扫描用新的并行版)
        CenteredScaleNorm.forward = old_csn
    elif impl == "old":
        CenteredScaleNorm.forward = old_csn
        O.logspace_scan = lambda nu, p, s0, chunk=64, p_min_hint=None: \
            old_gated_scan(p, (1.0 - p) * nu, s0, chunk, p_min_hint)
        L.logspace_scan = O.logspace_scan


def main():
    impl = sys.argv[1]
    T = int(sys.argv[2])
    B = int(sys.argv[3])
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 15

    torch.set_num_threads(threads)
    vocab = 8192
    torch.manual_seed(0)
    apply_impl(impl)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    ms = statistics.median(ts) * 1000
    q1, q3 = ts[len(ts) // 4], ts[(3 * len(ts)) // 4]
    print("RESULT " + json.dumps({
        "impl": impl, "T": T, "B": B, "threads": threads, "rounds": rounds,
        "step_ms": round(ms, 2), "min_ms": round(ts[0] * 1000, 2),
        "iqr_pct": round(100 * (q3 - q1) / ms, 1),
    }))


if __name__ == "__main__":
    main()