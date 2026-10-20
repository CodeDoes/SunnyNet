"""线程数与前后向耗时的独占复测 (更高重复数, 排除污染)."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402


def bench(fn, reps=15, warm=4):
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
    return med * 1000, iqr, ts[0] * 1000


def main():
    vocab, B, T = 8192, 8, 1024
    print("=== 线程数 (T=1024 B=8) — 前向 / 训练步 分开测 ===")
    print(f"{'threads':>8s}{'fwd_ms':>10s}{'fwd_IQR':>9s}"
          f"{'step_ms':>10s}{'step_IQR':>9s}{'min_step':>10s}")
    for th in (2, 4, 6, 8, 10):
        torch.set_num_threads(th)
        torch.manual_seed(0)
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

        f_ms, f_iqr, _ = bench(fwd, reps=11, warm=3)
        s_ms, s_iqr, s_min = bench(step, reps=11, warm=3)
        print(f"{th:>8d}{f_ms:>10.1f}{f_iqr:>9.1f}"
              f"{s_ms:>10.1f}{s_iqr:>9.1f}{s_min:>10.1f}")


if __name__ == "__main__":
    main()