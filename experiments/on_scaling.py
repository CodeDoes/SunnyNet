"""七架构前向延迟随序列长度的缩放实测.

回答一个此前未验证的问题: SunnyNet 的长序列优势(T>=834 交叉点, T=4096 快 12.6x)
是否同样适用于同复杂度的 O(N) 同行.

测量纪律(关键): 同一模型先在干净进程测, 再在"已构建过多个模型"的进程里测,
耗时相差 2 倍以上(堆碎片/分配器状态). 因此每个 (架构, 序列长度) 单元格
**单独起一个子进程**, 只构建一个模型, 避免跨单元格污染.

协议: 各架构计算栈参数量对齐到 SunnyNet small 目标(同 train.py 口径),
batch=1 前向, 重复 repeats 次取中位数. 机器独占时运行.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RES = ROOT / "experiments" / "results"
from sunnynet.linear_baselines import LINEAR_ARCHS  # noqa: E402

ARCHS = ["sunnynet", "transformer", *LINEAR_ARCHS]


def _cell(arch: str, L: int, threads: int, repeats: int, warmup: int,
          batch: int, backward: bool, tf_layers: int = 6):
    """子进程入口: 只构建一个模型并测量."""
    import statistics

    import torch

    from sunnynet.baseline import TransformerLM, solve_baseline_for_params
    from sunnynet.linear_baselines import BLOCKS, _LMBase, build_linear_baseline
    from sunnynet.model import SunnyConfig, SunnyNet

    vocab = 8192
    torch.set_num_threads(threads)
    ref = SunnyNet(SunnyConfig.small(vocab))
    target = ref.num_params(non_embedding=True)
    del ref

    if arch == "sunnynet":
        cfg = SunnyConfig.small(vocab)
        cfg.scan_chunk = 64
        model = SunnyNet(cfg)
    elif arch == "transformer":
        cfg, _, _ = solve_baseline_for_params(target, vocab, n_layers=tf_layers,
                                              n_heads=8, max_len=max(L, 4096))
        model = TransformerLM(cfg)
    else:
        cfg, _, _ = build_linear_baseline(arch, target, vocab,
                                          max_len=max(L, 4096), n_layers=4)
        model = _LMBase(cfg)
        for _ in range(cfg.n_layers):
            model.blocks.append(BLOCKS[arch](cfg))
        model._finish_init()

    model.train(backward)
    x = torch.randint(0, vocab, (batch, L))
    y = torch.randint(0, vocab, (batch, L))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3) if backward else None

    def run():
        if backward:
            lg = model(x)
            loss = torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                model(x)

    for _ in range(warmup):
        run()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    print(json.dumps({
        "arch": arch, "seq_len": L,
        "ms": round(med * 1000, 4),
        "ms_per_token": round(med * 1000 / (batch * L), 5),
        "min_ms": round(ts[0] * 1000, 4),
        "iqr_pct": round((ts[3 * len(ts) // 4] - ts[len(ts) // 4])
                         / max(1e-9, med) * 100, 2),
        "stack": model.num_params(non_embedding=True),
    }, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="256,512,1024,2048,4096")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--tf-layers", type=int, default=6)
    ap.add_argument("--archs", default=",".join(ARCHS))
    ap.add_argument("--out", default=str(RES / "on_scaling.json"))
    args = ap.parse_args()

    lens = [int(s) for s in args.lens.split(",")]
    archs = args.archs.split(",")

    out = {a: [] for a in archs}
    for L in lens:
        cells = []
        for arch in archs:
            cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                   "--_cell", arch, str(L), str(args.threads), str(args.repeats),
                   str(args.warmup), str(args.batch), str(int(args.backward)),
                   str(args.tf_layers)]
            r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not line:
                print(f"  !! {arch} T={L} 失败: {r.stderr[-400:]}")
                continue
            rec = json.loads(line[-1])
            out[arch].append(rec)
            cells.append(f"{arch}={rec['ms']:.1f}")
        print(f"  T={L:5d} " + " ".join(cells))

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n写出 {args.out}")

    tag = f"batch={args.batch}, {args.threads} 线程, 前向" + ("+反向" if args.backward else "")
    print(f"\n=== ms/token ({tag}) ===")
    print("arch".ljust(13) + "".join(f"T={L:<7d}" for L in lens))
    for arch in archs:
        print(arch.ljust(13) + "".join(
            f"{r['ms_per_token']:<9.4f}" for r in out[arch]))

    print("\n=== 相对 SunnyNet 的 ms/token 比值 (>1 = 比 SunnyNet 慢) ===")
    print("arch".ljust(13) + "".join(f"T={L:<7d}" for L in lens))
    base = {r["seq_len"]: r["ms_per_token"] for r in out["sunnynet"]}
    for arch in archs:
        print(arch.ljust(13) + "".join(
            f"{out[arch][i]['ms_per_token'] / base[L]:<9.2f}"
            for i, L in enumerate(lens)))

    print("\n=== IQR% (采样离散度) ===")
    print("arch".ljust(13) + "".join(f"T={L:<7d}" for L in lens))
    for arch in archs:
        print(arch.ljust(13) + "".join(f"{r['iqr_pct']:<9.1f}" for r in out[arch]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        _cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
              int(sys.argv[6]), int(sys.argv[7]), bool(int(sys.argv[8])),
              int(sys.argv[9]) if len(sys.argv) > 9 else 6)
    else:
        main()