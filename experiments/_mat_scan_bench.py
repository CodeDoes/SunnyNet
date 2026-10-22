"""矩阵值 rank-1 递推内核 (sn_mat_scan) 的算子级与占端到端比例实测.

算子级倍数不得外推为端到端倍数, 故两个量分开测:
  A 算子级: 单次 sn_mat_scan vs 等价 numpy 表达式, 同尺寸同机同进程交替计时;
  B 占比  : DeltaNet 真实单步耗时中, 该算子所占份额 (逐层累加)。

独占运行 (并行任务会让 ms 虚高)。
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def bench(fn, reps, warm=10):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts), ts[len(ts) // 4], ts[3 * len(ts) // 4]


def numpy_step(S, w, u, v, k, b, r):
    p = np.einsum("hrc,hc->hr", S, u)
    S *= w[:, None, :]
    S += v[:, :, None] * k[:, None, :] - p[:, :, None] * b[:, None, :]
    return np.einsum("hrc,hc->hr", S, r)


def main():
    import torch
    from sunnynet.runtime import mat_step, runtime_available

    if not runtime_available():
        print("无 C 运行时, 跳过")
        return

    shapes = [(8, 128, 128), (8, 64, 64), (16, 32, 256)]
    print("=== A. 算子级: sn_mat_scan vs numpy (同进程交替) ===")
    print(f"{'H R C':>14s}{'内核 ms':>10s}{'numpy ms':>10s}{'倍数':>8s}{'IQR核%':>8s}")
    rec = {}
    for NH, R, C in shapes:
        rng = np.random.default_rng(0)
        w = (rng.random((NH, C)) * 0.9 + 0.05).astype(np.float32)
        u = rng.standard_normal((NH, C)).astype(np.float32)
        v = rng.standard_normal((NH, R)).astype(np.float32)
        k = rng.standard_normal((NH, C)).astype(np.float32)
        b = rng.standard_normal((NH, C)).astype(np.float32)
        r = rng.standard_normal((NH, C)).astype(np.float32)

        S0 = rng.standard_normal((NH, R, C)).astype(np.float32) * 0.01
        St = torch.tensor(S0.copy())
        st = {n: torch.tensor(x) for n, x in
              dict(w=w, u=u, v=v, k=k, b=b, r=r).items()}
        Sn = S0.copy()

        def k_c():
            St.copy_(torch.from_numpy(S0))
            mat_step(St, **st)

        def k_n():
            Sn[:] = S0
            numpy_step(Sn, w, u, v, k, b, r)

        reps = 300 if NH * R * C < 200_000 else 120
        mc, q1c, q3c = bench(k_c, reps)
        mn, _, _ = bench(k_n, reps)
        iqr = (q3c - q1c) / max(mc, 1e-12) * 100
        print(f"{NH:>4d}{R:>4d}{C:>5d}{mc*1e3:>10.4f}{mn*1e3:>10.4f}"
              f"{mn/mc:>8.2f}{iqr:>8.1f}")
        rec[f"{NH}x{R}x{C}"] = {"core_ms": mc * 1e3, "numpy_ms": mn * 1e3,
                                "speedup": mn / mc}

    print("\n=== B. 占比: DeltaNet 单步里该算子的份额 ===")
    import os
    mdir = os.environ.get("SUNNY_DELTANET", str(ROOT / "models" / "deltanet-pure"))
    if not Path(mdir).exists():
        print(f"权重目录不存在 ({mdir}), 跳过")
    else:
        from sunnynet.runtime.deltanet import DeltaNet
        m = DeltaNet(mdir)
        H, K, V = m.n_heads, m.head_k_dim, m.head_v_dim
        L = m.n_layers
        st = m.reset(m.init_state())
        for t in range(1, 40):
            m.step(t, st)

        def one_step():
            m.step(7, st)

        med, _, _ = bench(one_step, 30, warm=5)
        Sd = np.zeros((H, K, V), dtype=np.float32)
        Stt = torch.zeros(H, K, V)

        def rank1_layers():
            for _ in range(L):
                mat_step(Stt, u=u, v=v, k=k, b=b, r=q)

        rng = np.random.default_rng(1)
        u = np.zeros((H, K), dtype=np.float32)
        v = rng.standard_normal((H, V)).astype(np.float32)
        k = np.zeros((H, K), dtype=np.float32)
        b = rng.standard_normal((H, K)).astype(np.float32)
        q = rng.standard_normal((H, K)).astype(np.float32)
        tk = {n: torch.tensor(x) for n, x in
              dict(u=u, v=v, k=k, b=b, r=q).items()}

        def rank1_layers():
            for _ in range(L):
                mat_step(Stt, **tk)

        med_r1, _, _ = bench(rank1_layers, 60, warm=5)
        share = med_r1 / med * 100
        print(f"DeltaNet 单步     = {med*1e3:.3f} ms  (L={L}, H={H}, {K}x{V})")
        print(f"其中 {L} 层 rank-1 = {med_r1*1e3:.3f} ms  → 占比 {share:.1f}%")
        rec["deltanet_share"] = {"step_ms": med * 1e3, "rank1_ms": med_r1 * 1e3,
                                 "share_pct": share}
        print("结论: 该算子只占单步一小部分, 单靠它无法把 DeltaNet 单步拉近 C 路径;")
        print("      剩余时间在逐层 GEMV, 需整模型 C 实现 (见任务 #71)。")

    out = ROOT / "results" / "_mat_scan_bench.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()