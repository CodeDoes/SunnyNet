"""当前架构单步训练的分段耗时拆分 (独占运行).

把 block 内计算拆成: 归一化/卷积 -> _core(投影+竞争+门控) -> 扫描 -> _emit(读出),
并单独测前向 vs 反向, 定位训练慢在哪一段.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import logspace_scan  # noqa: E402


def bench(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    iqr = (ts[3 * len(ts) // 4] - ts[len(ts) // 4]) / max(med, 1e-12) * 100
    return med * 1000, iqr


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    vocab, B, T = 8192, 8, 1024
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def fwd():
        return m(x)

    def full():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    f_ms, f_iqr = bench(fwd)
    b_ms, b_iqr = bench(full)
    print(f"前向           {f_ms:9.1f} ms  (IQR {f_iqr:.1f}%)")
    print(f"前向+反向+更新 {b_ms:9.1f} ms  (IQR {b_iqr:.1f}%)")
    print(f"反向占比       {(b_ms - f_ms) / b_ms * 100:9.1f} %")

    # 扫描单独计时 (4 层, 每层一次)
    blk = m.blocks[0]
    s0 = torch.zeros(B, blk.dim)
    a = torch.randn(B, T, blk.dim)
    u, gate, p = blk._core(a)
    sc_ms, _ = bench(lambda: logspace_scan(u, p, s0, chunk=64,
                                           p_min_hint=blk.p_min_bound))
    print(f"\n单层扫描       {sc_ms:9.2f} ms  x{len(m.blocks)} 层 = "
          f"{sc_ms * len(m.blocks):.1f} ms")
    print(f"扫描占前向     {sc_ms * len(m.blocks) / f_ms * 100:9.1f} %")

    feats = blk.norm_in(x.float() if x.dtype.is_floating_point else
                        m.embed(x))
    raw = blk.norm_in(m.embed(x))
    core_ms, _ = bench(lambda: blk._core(blk._apply_local(raw, torch.zeros(
        B, blk.taps - 1, blk.dim))))
    print(f"单层 _core     {core_ms:9.2f} ms  x{len(m.blocks)} = "
          f"{core_ms * len(m.blocks):.1f} ms  "
          f"({core_ms * len(m.blocks) / f_ms * 100:.1f}% 前向)")
    del feats

    st, _ = logspace_scan(u, p, s0, chunk=64, p_min_hint=blk.p_min_bound)
    emit_ms, _ = bench(lambda: blk._emit(u, st, s0))
    print(f"单层 _emit     {emit_ms:9.2f} ms  x{len(m.blocks)} = "
          f"{emit_ms * len(m.blocks):.1f} ms  "
          f"({emit_ms * len(m.blocks) / f_ms * 100:.1f}% 前向)")


if __name__ == "__main__":
    main()