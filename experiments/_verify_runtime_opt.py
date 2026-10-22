"""验证行主序 + 并行的 C runtime 与 torch 前向数值一致, 并测加速比。

同一 checkpoint: torch 逐 token 前向 vs C runtime sn_step。
口径: 三个指标 (logits max|diff|、argmax、末尾 token), 多轮交错取中位。
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
PROMPT = [1] + [100 + (i * 37) % 4000 for i in range(31)]

# 与 tests/test_runtime_model.py 同一口径: forward_step + 单线程 torch
torch.set_num_threads(1)


def main():
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, _load

    m, cfg, _ = load_model(str(CKPT))
    m.eval()
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)

    # torch 参照: forward_step 单步, 与 C sn_step 对应
    ref = []
    with torch.no_grad():
        st = m.init_state(1)
        for t in PROMPT:
            lg, st = m.forward_step(torch.tensor([t]), st)
            ref.append(lg[0].float().numpy())
    ref = np.stack(ref)

    h = SnModel(BLOB, d["vocab_size"], d["dim"], d["inner"], d["rank"],
                d.get("taps", 3), d["layers"], d.get("group", 8), 1)
    h.reset()
    got = []
    for t in PROMPT:
        got.append(np.asarray(h.step(t), dtype=np.float32))
    got = np.stack(got)

    diff = float(np.abs(ref - got).max())
    am_ref = ref.argmax(-1)
    am_got = got.argmax(-1)
    print(f"torch vs C   n={len(PROMPT)}")
    print(f"  logits max|diff| = {diff:.3e}")
    print(f"  argmax 一致      = {int((am_ref == am_got).sum())}/{len(PROMPT)}")
    print(f"  末尾 token       = torch {int(am_ref[-1])} / C {int(am_got[-1])}")

    # 单步耗时: 多轮交错 head-to-head
    lib = _load()
    reps = 5
    ts_c, ts_t = [], []
    for r in range(reps):
        h.reset()
        t0 = time.perf_counter()
        for t in PROMPT:
            h.step(t)
        ts_c.append((time.perf_counter() - t0) / len(PROMPT))
        h.reset()
        with torch.no_grad():
            t0 = time.perf_counter()
            st = m.init_state(1)
            for t in PROMPT:
                _, st = m.forward_step(torch.tensor([t]), st)
            ts_t.append((time.perf_counter() - t0) / len(PROMPT))
    mc, mt = statistics.median(ts_c) * 1e3, statistics.median(ts_t) * 1e3
    wins = sum(a < b for a, b in zip(ts_c, ts_t))
    print(f"\n单步 (ms/token, {reps} 轮交错)")
    print(f"  C runtime 中位 {mc:8.2f}   min {min(ts_c)*1e3:8.2f}")
    print(f"  torch     中位 {mt:8.2f}   min {min(ts_t)*1e3:8.2f}")
    print(f"  head-to-head C 胜 {wins}/{reps}   倍数 {mt/mc:.2f}x")
    print(f"  C 有效带宽 {BLOB.stat().st_size/(mc/1e3)/1e9:.2f} GB/s")
    print(f"  OMP 线程数 {lib.sn_get_threads()}")


if __name__ == "__main__":
    main()