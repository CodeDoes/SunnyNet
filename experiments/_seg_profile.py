"""单块分段开销剖析: 定位 SunnyNet 训练速度的真实构成.

运行: python -X utf8 experiments/_seg_profile.py [T] [B] [threads]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import logspace_scan  # noqa: E402


def bench(fn, reps=21, warm=5):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    vocab, B, T = 8192, args.batch, args.seq_len
    m = SunnyNet(SunnyConfig.small(vocab))
    blk = m.blocks[0]
    print(f"dim={blk.dim} inner={blk.inner} taps={blk.taps} group={blk.group} "
          f"chunk={blk.scan_chunk} p_min_bound={blk.p_min_bound}")

    x = torch.randn(B, T, blk.dim)
    s = torch.zeros(B, blk.dim)
    hist = torch.zeros(B, blk.taps - 1, blk.dim)

    def s_norm():
        return blk.norm_in(x)

    r = blk.norm_in(x)

    def s_local():
        return blk._apply_local(blk.norm_in(x), hist)

    a = blk._apply_local(r, hist)

    def s_core():
        return blk._core(blk._apply_local(blk.norm_in(x), hist))

    u, gate, p = blk._core(a)

    def s_scan():
        return logspace_scan(u, p, s, chunk=blk.scan_chunk, p_min_hint=blk.p_min_bound)

    states, _ = logspace_scan(u, p, s, chunk=blk.scan_chunk, p_min_hint=blk.p_min_bound)

    def s_emit():
        return blk._emit(u, states, s)

    def s_feedback():
        return blk._feedback(states, s)

    def s_readout():
        return blk._readout(u, None)

    def s_compete():
        return blk._compete(gate)

    def s_delta():
        return blk.delta()

    cells = [
        ("norm_in", s_norm), ("_apply_local(conv)", s_local),
        ("_core(V/U/compete/p)", s_core), ("_scan(logspace_scan)", s_scan),
        ("_readout", s_readout), ("_feedback(circ)", s_feedback),
        ("_emit(full)", s_emit), ("_compete", s_compete), ("delta", s_delta),
    ]
    print(f"\n{'段':26s}{'ms':>9s}{'min':>9s}")
    base = {}
    for n, f in cells:
        md, mn = bench(f)
        base[n] = md
        print(f"{n:26s}{md:>9.3f}{mn:>9.3f}")

    full, fullmin = bench(lambda: blk(x, s, hist))
    print(f"\n{'单块整段':26s}{full:>9.3f}{fullmin:>9.3f}")
    print(f"× 4 层 = {full*4:.2f} ms (前向); 反向约 1.6-2.2×")

    print(f"\n扫描串行块数 = T/chunk = {T//blk.scan_chunk}, 每前向 {T//blk.scan_chunk*4} 次")

    # 线性层参数统计
    n_lin = sum(1 for _ in blk.modules() if isinstance(_, torch.nn.Linear))
    print(f"块内 nn.Linear 数 = {n_lin}")


if __name__ == "__main__":
    main()