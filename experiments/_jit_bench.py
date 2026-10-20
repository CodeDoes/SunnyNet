"""torch.jit.script 对扫描的效果实测 (本机唯一可用的编译后端).

扫描有 ceil(T/chunk) 次 Python 循环, JIT 可尝试消除其调度开销.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


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


def main():
    torch.set_num_threads(8)
    from sunnynet.ops import gated_scan

    B, T, N = 8, 1024, 512
    p = (torch.rand(B, T, N) * 0.5 + 0.5).clamp(max=0.999)
    g = torch.randn(B, T, N) * 0.1
    s0 = torch.zeros(B, N)

    ms, mn = bench(lambda: gated_scan(p, g, s0, chunk=64, p_min_hint=0.5))
    print(f"eager gated_scan        {ms:8.2f} ms (min {mn:7.2f})")

    # JIT 版本 (固定形状, 无 python 级可变参数)
    @torch.jit.script
    def gated_scan_jit(p: torch.Tensor, g: torch.Tensor, s0: torch.Tensor,
                       chunk: int) -> torch.Tensor:
        Tn = p.shape[-2]
        s = s0
        outs = torch.jit.annotate(list, [])
        for start in range(0, Tn, chunk):
            end = start + chunk
            if end > Tn:
                end = Tn
            lp = torch.log(p[:, start:end]).transpose(-2, -1).contiguous()
            gg = g[:, start:end].transpose(-2, -1).contiguous()
            P = torch.cumsum(lp, dim=-1)
            A = torch.cumsum(gg * torch.exp(-P), dim=-1)
            s_c = (torch.exp(P) * (s.unsqueeze(-1) + A)).transpose(-2, -1)
            outs.append(s_c)
            s = s_c[..., -1, :]
        return torch.cat(outs, dim=-2)

    try:
        r1 = gated_scan_jit(p, g, s0, 64)
        ref = gated_scan(p, g, s0, chunk=64, p_min_hint=0.5)[0]
        print(f"JIT 数值差 max|Δ| = {(r1 - ref).abs().max().item():.3e}")
        ms2, mn2 = bench(lambda: gated_scan_jit(p, g, s0, 64))
        print(f"jit   gated_scan        {ms2:8.2f} ms (min {mn2:7.2f})"
              f"   变化 {(ms2 - ms) / ms * 100:+.1f}%")
    except Exception as e:
        print(f"JIT 编译失败: {type(e).__name__}: {str(e)[:200]}")

    # 整模型 jit.script
    from sunnynet.model import SunnyConfig, SunnyNet
    m = SunnyNet(SunnyConfig.small(8192)).train()
    x = torch.randint(0, 8192, (B, T))
    try:
        mj = torch.jit.script(m)
        with torch.no_grad():
            r = mj(x)
        with torch.no_grad():
            ref = m(x)
        print(f"\n整模型 JIT max|Δ| = {(r - ref).abs().max().item():.3e}")
        ms3, _ = bench(lambda: mj(x))
        ms4, _ = bench(lambda: m(x))
        print(f"整模型 jit  {ms3:8.2f} ms  vs eager {ms4:8.2f} ms"
              f"   变化 {(ms3 - ms4) / ms4 * 100:+.1f}%")
    except Exception as e:
        print(f"\n整模型 JIT 失败: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()