"""扫描实现 A/B: 原版(dim=-2) vs 转置版(dim=-1). 每个配置独立子进程.

用法: python _ab_scan.py <orig|trans> <T> <B>
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
            p_min = p.detach().amin().item() if p_min_hint is None else p_min_hint
            chunk = ops._safe_chunk(p_min, chunk)
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

        ops.gated_scan = gated_scan_orig

        def logspace_scan(nu, p, s0, chunk=64, p_min_hint=None):
            return gated_scan_orig(p, (1.0 - p) * nu, s0, chunk=chunk,
                                   p_min_hint=p_min_hint)
        ops.logspace_scan = logspace_scan
        import sunnynet.layers as _l
        _l.logspace_scan = logspace_scan

    torch.manual_seed(0)
    vocab = 8192
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def fwd():
        with torch.no_grad():
            m(x)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def bench(fn, reps=15, warm=4):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return statistics.median(ts) * 1000, ts[0] * 1000

    f_med, f_min = bench(fwd)
    s_med, s_min = bench(step, reps=11, warm=3)
    print(f"RESULT mode={mode} T={T} B={B} "
          f"fwd_med={f_med:.1f} fwd_min={f_min:.1f} "
          f"step_med={s_med:.1f} step_min={s_min:.1f}")


if __name__ == "__main__":
    main()