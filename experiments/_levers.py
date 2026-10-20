"""候选优化杠杆的独占实测.

逐项验证: 线程数 / 优化器 foreach / 元素级算子数 / bf16 可用性.
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


def bench(step_fn, reps=9, warm=3):
    for _ in range(warm):
        step_fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step_fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    iqr = (ts[3 * len(ts) // 4] - ts[len(ts) // 4]) / max(med, 1e-12) * 100
    return med * 1000, iqr


def main():
    vocab, B, T = 8192, 8, 1024
    print("=== A. 线程数缩放 (T=1024, B=8, fwd+bwd) ===")
    print(f"{'threads':>8s}{'ms':>10s}{'IQR%':>7s}{'相对8线程':>10s}")
    base = None
    for th in (1, 2, 4, 8, 12, 16):
        torch.set_num_threads(th)
        torch.manual_seed(0)
        m = SunnyNet(SunnyConfig.small(vocab)).train()
        x = torch.randint(0, vocab, (B, T))
        y = torch.randint(0, vocab, (B, T))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

        def step():
            lg = m(x)
            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        ms, iqr = bench(step, reps=7, warm=2)
        if th == 8:
            base = ms
        print(f"{th:>8d}{ms:>10.1f}{iqr:>7.1f}")

    print("\n=== B. AdamW foreach 开关 (8 线程) ===")
    torch.set_num_threads(8)
    for fe in (False, True):
        torch.manual_seed(0)
        m = SunnyNet(SunnyConfig.small(vocab)).train()
        x = torch.randint(0, vocab, (B, T))
        y = torch.randint(0, vocab, (B, T))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=fe)

        def step(fe=fe):
            lg = m(x)
            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        ms, iqr = bench(step, reps=7, warm=2)
        print(f"  foreach={str(fe):5s}  {ms:8.1f} ms  (IQR {iqr:.1f}%)")

    print("\n=== C. CPU 指令集能力 (决定 bf16/融合是否可行) ===")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  mkldnn 可用     = {torch.backends.mkldnn.is_available()}")
    print(f"  bf16 支持       = {torch.cpu._is_avx512_bf16_supported()}"
          if hasattr(torch.cpu, "_is_avx512_bf16_supported") else "  bf16 API 缺失")
    caps = torch.backends.cpu.get_cpu_capability()
    print(f"  cpu capability  = {caps}")
    try:
        amx = torch.backends.cpu._get_cpu_capability()
    except Exception:
        amx = None


if __name__ == "__main__":
    main()