"""转置扫描对"反向"的影响单测: 判断该优化对训练是否净正收益.

前向 A/B 显示转置版快 6-8%, 但训练步无收益 -> 怀疑反向因多次
transpose+contiguous 产生额外拷贝. 本脚本单测反向.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main():
    mode, T, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    torch.set_num_threads(8)

    import sunnynet.ops as ops
    from sunnynet.model import SunnyConfig, SunnyNet

    if mode == "orig":
        def gated_scan_orig(p, g, s0, chunk=64, p_min_hint=None):
            p, g = torch.broadcast_tensors(p, g)
            Tn = p.shape[-2]
            lp = torch.log(p)
            chunk = ops._safe_chunk(0.5, chunk)
            s = s0
            outs = []
            for start in range(0, Tn, chunk):
                end = min(start + chunk, Tn)
                P = torch.cumsum(lp[..., start:end, :], dim=-2)
                A = torch.cumsum(g[..., start:end, :] * torch.exp(-P), dim=-2)
                s_c = torch.exp(P) * (s.unsqueeze(-2) + A)
                outs.append(s_c)
                s = s_c[..., -1, :]
            return torch.cat(outs, dim=-2), s

        def logspace_scan(nu, p, s0, chunk=64, p_min_hint=None):
            return gated_scan_orig(p, (1.0 - p) * nu, s0, chunk=chunk,
                                   p_min_hint=p_min_hint)
        ops.gated_scan = gated_scan_orig
        ops.logspace_scan = logspace_scan
        import sunnynet.layers as _l
        _l.logspace_scan = logspace_scan

    torch.manual_seed(0)
    vocab = 8192
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def bwd_only():
        opt.zero_grad(set_to_none=True)
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        return loss

    with torch.no_grad():
        m(x)
    # 先构建一次图
    bwd_only()

    ts = []
    for _ in range(13):
        t0 = time.perf_counter()
        bwd_only()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts) * 1000
    print(f"RESULT mode={mode} T={T} B={B} bwd_med={med:.1f} bwd_min={ts[0]*1000:.1f}")


if __name__ == "__main__":
    main()