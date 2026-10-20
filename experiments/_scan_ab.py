"""扫描实现对照: 逐块串行 vs 分块并行.

逐块串行 (现行): 每块一次 Python 循环 + 5 个算子, T=1024 需 16 次, T=4096 需 64 次.
分块并行 (候选): 块内一次 cumsum 覆盖全部块, 块间串行仅 nblk 步.

运行: python -X utf8 experiments/_scan_ab.py equiv
      python -X utf8 experiments/_scan_ab.py bench orig|para [T] [B] [N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.ops import _safe_chunk, gated_scan  # noqa: E402


def orig_scan(p, g, s0, chunk=64, p_min_hint=None):
    """现行逐块串行扫描 (gated_scan 主体, 便于同函数内对照)."""
    p, g = torch.broadcast_tensors(p, g)
    T = p.shape[-2]
    lp = torch.log(p)
    p_min = p.detach().amin().item() if p_min_hint is None else p_min_hint
    ch = _safe_chunk(p_min, chunk)
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


def para_scan(p, g, s0, chunk=64, p_min_hint=None):
    """分块并行扫描: 块内并行前缀, 块间串行 nblk 步, 状态由 s_in 修正."""
    p, g = torch.broadcast_tensors(p, g)
    T = p.shape[-2]
    p_min = p.detach().amin().item() if p_min_hint is None else p_min_hint
    ch = _safe_chunk(p_min, chunk)
    nblk = (T + ch - 1) // ch
    pad = nblk * ch - T
    lead = p.shape[:-2]
    N = p.shape[-1]
    if pad:
        p = torch.cat([p, p.new_ones(*lead, pad, N)], dim=-2)
        g = torch.cat([g, g.new_zeros(*lead, pad, N)], dim=-2)
    lp = torch.log(p).reshape(*lead, nblk, ch, N)
    gg = g.reshape(*lead, nblk, ch, N)
    P = torch.cumsum(lp, dim=-2)
    E = torch.exp(P)
    A = torch.cumsum(gg * torch.exp(-P), dim=-2)
    Elast = E[..., :, -1, :]
    Alast = A[..., :, -1, :]
    s = s0
    s_ins = []
    for c in range(nblk):
        s_ins.append(s)
        s = Elast[..., c, :] * (s + Alast[..., c, :])
    s_in_seq = torch.stack(s_ins, dim=-2)
    S = E * (s_in_seq.unsqueeze(-2) + A)
    S = S.reshape(*lead, nblk * ch, N)[..., :T, :]
    return S, s


def cmd_equiv(args):
    torch.manual_seed(0)
    B, T, N = 2, 384, 48
    for dt in (torch.float64, torch.float32):
        p = (torch.rand(B, T, N, dtype=dt) * 0.4 + 0.5)
        g = torch.randn(B, T, N, dtype=dt) * 0.1
        s0 = torch.randn(B, N, dtype=dt) * 0.1
        r1, e1 = orig_scan(p, g, s0, 64, 0.5)
        r2, e2 = para_scan(p, g, s0, 64, 0.5)
        d = (r1 - r2).abs().max().item()
        de = (e1 - e2).abs().max().item()
        ref = r1.abs().max().item()
        print(f"{str(dt):18s} max|dS|={d:.3e}  rel={d/ref:.3e}  max|dS_T|={de:.3e}")

    # 逐 token 递推校验
    s = s0.clone()
    for t in range(T):
        s = p[:, t] * s + g[:, t]
    r2, e2 = para_scan(p, g, s0, 64, 0.5)
    print(f"{'vs 逐 token 递推':18s} max|dS_T|={(s - e2).abs().max().item():.3e}")


def cmd_bench(args):
    torch.set_num_threads(args.threads)
    B, T, N = args.batch, args.seq_len, args.n_dim
    p = torch.rand(B, T, N) * 0.4 + 0.5
    g = torch.randn(B, T, N) * 0.1
    s0 = torch.zeros(B, N)

    fn = orig_scan if args.impl == "orig" else para_scan

    def run():
        return fn(p, g, s0, 64, 0.5)

    for _ in range(5):
        run()
    ts = []
    for _ in range(21):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    ms = statistics.median(ts) * 1000
    nblk = T // _safe_chunk(0.5, 64)
    print("RESULT " + json.dumps({
        "impl": args.impl, "T": T, "B": B, "N": N,
        "ms": round(ms, 4), "min_ms": round(ts[0] * 1000, 4),
        "nblk": nblk, "threads": args.threads,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "bench"])
    ap.add_argument("impl", nargs="?", default="para")
    ap.add_argument("seq_len", nargs="?", type=int, default=1024)
    ap.add_argument("batch", nargs="?", type=int, default=8)
    ap.add_argument("n_dim", nargs="?", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "equiv":
        cmd_equiv(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()