"""Runtime 通用递推内核 vs PyTorch: CPU 速度对照.

三条基线, 必须分开读:
  A. torch 逐 token Python 循环  —— 参照下限, 不代表任何真实部署方式
  B. torch 向量化分块扫描 (ops.gated_scan, cumsum 闭式解) —— 真实竞争基线
  C. Runtime 专用内核 (AVX2/FMA, 单趟)

口径: 每格独立子进程; 单线程; 10 次取中位; B=1, N 为通道维。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results" / "_runtime_bench.json"
PY = sys.executable

CASES = [
    (1, 256, 128), (1, 512, 128), (1, 1024, 128), (1, 4096, 128),
    (1, 1024, 320), (1, 4096, 640),
]


def child(B, T, N, gated):
    import time

    import torch
    sys.path.insert(0, str(ROOT))
    from sunnynet.runtime import load_runtime, scan
    from sunnynet.ops import gated_scan
    lib = load_runtime()
    lib.sn_set_threads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    a = torch.rand(B, T, N) * 0.5 + 0.5
    b = torch.randn(B, T, N)
    s0 = torch.zeros(B, N)

    def median(fn, warm=2, reps=10):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return sorted(ts)[len(ts) // 2]

    # A: python 循环
    def loop():
        s = s0.clone()
        store = torch.empty(B, T, N)
        for t in range(T):
            at = a[:, t]
            bt = (1.0 - at) * b[:, t] if gated else b[:, t]
            s = at * s + bt
            store[:, t] = s
        return store

    # B: torch 向量化分块扫描, 同一数学
    def vec():
        return gated_scan(a, (1.0 - a) * b if gated else b, s0)[0]

    # C: 内核
    def kern():
        return scan(a, b, s0, gated=gated)[0]

    t_loop = median(loop)
    t_vec = median(vec)
    t_kern = median(kern)

    # 三分支数值一致性
    r_loop, r_vec, r_kern = loop(), vec(), kern()
    d_loop = (r_kern - r_loop).abs().max().item()
    d_vec = (r_kern - r_vec).abs().max().item()
    return {"B": B, "T": T, "N": N, "gated": gated,
            "loop_ms": t_loop, "vec_ms": t_vec, "kernel_ms": t_kern,
            "vs_loop": t_loop / t_kern, "vs_vec": t_vec / t_kern,
            "diff_loop": d_loop, "diff_vec": d_vec}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        B, T, N, g = map(int, sys.argv[2:6])
        print(json.dumps(child(B, T, N, bool(g))))
        return 0
    rows = []
    for B, T, N in CASES:
        for gated in (0, 1):
            r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                "--child", str(B), str(T), str(N), str(gated)],
                               capture_output=True, text=True, cwd=str(ROOT))
            try:
                rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
            except Exception:
                print(f"  失败 T={T} N={N} gated={gated}: {(r.stderr or '')[-200:]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'T':>5} {'N':>4} {'模式':>6} {'循环ms':>9} {'向量ms':>9} {'内核ms':>9} "
          f"{'vs循环':>8} {'vs向量':>8}")
    print("-" * 62)
    for d in rows:
        print(f"{d['T']:>5} {d['N']:>4} {'GATED' if d['gated'] else 'FMA':>6} "
              f"{d['loop_ms']:>9.3f} {d['vec_ms']:>9.3f} {d['kernel_ms']:>9.3f} "
              f"{d['vs_loop']:>7.1f}x {d['vs_vec']:>7.2f}x")
    print("-" * 62)
    vs_vec = [d["vs_vec"] for d in rows]
    print(f"对向量化基线: 中位 {sorted(vs_vec)[len(vs_vec)//2]:.2f}x, "
          f"范围 {min(vs_vec):.2f}-{max(vs_vec):.2f}x")
    md = max(max(d["diff_loop"], d["diff_vec"]) for d in rows)
    print(f"三分支最大数值差 {md:.2e}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())