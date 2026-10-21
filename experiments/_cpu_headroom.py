"""CPU 提速空间探测: 矩阵峰值 / 低精度 / INT8 / 线程扩展.

每格独立子进程 (--cell), 避免同进程多次构建污染计时.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bench(fn, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def cell_matmul(n: int, dtype: str) -> dict:
    torch.set_num_threads(torch.get_num_threads())
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    if dtype == "fp32":
        A, B = a, b
    elif dtype == "bf16":
        A, B = a.bfloat16(), b.bfloat16()
    elif dtype == "fp16":
        A, B = a.half(), b.half()
    else:
        raise ValueError(dtype)
    t = _bench(lambda: A @ B)
    flops = 2 * n ** 3
    return {"cell": f"matmul_{dtype}_{n}", "seconds": t, "gflops": flops / t / 1e9}


def cell_int8(m: int, n: int, k: int) -> dict:
    a = torch.randint(-128, 127, (m, k), dtype=torch.int8)
    b = torch.randint(-128, 127, (k, n), dtype=torch.int8)
    try:
        t = _bench(lambda: torch._int_mm(a, b))
    except Exception as e:
        return {"cell": f"int8_{m}x{n}x{k}", "error": f"{type(e).__name__}: {e}"}
    flops = 2 * m * n * k
    out = torch._int_mm(a, b)
    return {"cell": f"int8_{m}x{n}x{k}", "seconds": t,
            "gflops": flops / t / 1e9, "out_dtype": str(out.dtype)}


def cell_threads(n: int, threads: int) -> dict:
    torch.set_num_threads(threads)
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    t = _bench(lambda: a @ b)
    return {"cell": f"threads_{threads}_{n}", "seconds": t,
            "gflops": 2 * n ** 3 / t / 1e9}


def cell_fusion_chain(batch: int, dim: int, layers: int) -> dict:
    """逐元素算子链: 分离 vs 就地/融合的端到端差异."""
    x = torch.randn(batch, dim, dim)

    def separate():
        h = x
        for _ in range(layers):
            g = torch.sigmoid(h)
            h = torch.sigmoid(g - g.roll(1, dims=-1)) * h
            h = h * (1.0 - g) + g
        return h

    def inplace():
        h = x.clone()
        for _ in range(layers):
            g = torch.sigmoid(h)
            h = torch.sigmoid(g - g.roll(1, dims=-1)).mul_(h)
            h.addcmul_(g, 1.0 - g)
        return h

    t_sep = _bench(separate)
    t_inp = _bench(inplace)
    return {"cell": f"fusion_chain_{batch}x{dim}x{layers}", "separate": t_sep,
            "inplace": t_inp, "ratio": t_sep / t_inp,
            "maxdiff": float((separate() - inplace()).abs().max())}


def cell_scan_dtype(T: int, n: int, dtype: str) -> dict:
    """扫描累积对数值精度的敏感度 (决定能否低精度/量化)."""
    from sunnynet.ops import logspace_scan
    torch.manual_seed(0)
    nu64 = torch.randn(2, T, n, dtype=torch.float64)
    p64 = torch.linspace(0.9, 0.999, T, dtype=torch.float64).view(1, T, 1).expand(2, T, n).contiguous()
    s064 = torch.zeros(2, n, dtype=torch.float64)
    ref, _ = logspace_scan(nu64, p64, s064)
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    got, _ = logspace_scan(nu64.to(dt), p64.to(dt), s064.to(dt))
    ref_max = float(ref.abs().max())
    return {"cell": f"scan_{dtype}_T{T}",
            "rel": float((got.double() - ref).abs().max() / ref_max)}


def cell_small(m: int, k: int, n: int, threads: int, batch: int = 1) -> dict:
    """实际训练尺寸的 GEMM (batch 可 >1 表示 B*T 折叠)."""
    torch.set_num_threads(threads)
    M = m * batch
    a = torch.randn(M, k)
    b = torch.randn(k, n)
    t = _bench(lambda: a @ b)
    return {"cell": f"small_{M}x{k}x{n}_t{threads}", "seconds": t,
            "gflops": 2 * M * k * n / t / 1e9}


def cell_ops_chain(batch: int, seq: int, dim: int, inner: int, layers: int) -> dict:
    """模拟一个块内的逐元素算子开销 vs GEMM 开销占比."""
    x = torch.randn(batch * seq, dim)
    w1 = torch.randn(dim, inner)
    w2 = torch.randn(inner, dim)

    def gemm_path():
        h = x @ w1
        h = torch.sigmoid(h) * h
        return h @ w2

    def elem_path():
        h = x
        for _ in range(layers):
            h = torch.sigmoid(h)
            h = h * (1.0 - h)
            h = h - h.mean(dim=-1, keepdim=True)
        return h

    t_g = _bench(gemm_path)
    t_e = _bench(elem_path)
    return {"cell": f"ops_{batch}x{seq}_d{dim}_i{inner}_L{layers}",
            "gemm": t_g, "elem": t_e, "elem_over_gemm": t_e / t_g}


CELLS = {
    "matmul": cell_matmul,
    "int8": cell_int8,
    "threads": cell_threads,
    "fusion": cell_fusion_chain,
    "scan": cell_scan_dtype,
    "small": cell_small,
    "ops": cell_ops_chain,
}


def run_isolated(cells: list[tuple[str, dict]]) -> list[dict]:
    out = []
    for name, kw in cells:
        code = (f"import sys; sys.path.insert(0, r'{ROOT}');"
                f"import experiments._cpu_headroom as M, json;"
                f"print(json.dumps(M.CELLS['{name}'](**{kw!r}), ensure_ascii=False))")
        r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                           capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            out.append({"cell": f"{name}{kw}", "error": r.stderr.strip()[-300:]})
            continue
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        out.append(json.loads(line[-1]) if line else {"cell": f"{name}{kw}", "error": "no output"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cell:
        name, kw = args.cell.split(":", 1)
        print(json.dumps(CELLS[name](**json.loads(kw))))
        return

    print(f"threads={torch.get_num_threads()} capability={torch.backends.cpu.get_cpu_capability()}")
    cells = []
    for n in (256, 512, 1024, 2048):
        cells += [("matmul", {"n": n, "dtype": "fp32"}),
                  ("matmul", {"n": n, "dtype": "bf16"})]
    cells += [("threads", {"n": 1024, "threads": t}) for t in (1, 2, 4, 6, 8, 12)]
    cells += [("int8", {"m": 1024, "n": 1024, "k": 1024}),
              ("int8", {"m": 512, "n": 512, "k": 512})]
    res = run_isolated(cells)
    for r in res:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()