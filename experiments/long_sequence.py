"""长序列对照: 验证 O(N) 优势随序列长度增长.

同一 token 预算下, 在两个架构上分别跑 T in {256, 512, 1024}, 记录
每 token 耗时 / 吞吐 / 峰值内存 / 验证损失. 用于检验"线性复杂度在长序列
上带来实际收益"这一主张 —— 若收益不随 T 增长, 该主张不成立.

同时产出前向延迟 vs 序列长度数据 (experiments/results/seq_scaling.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
RES = ROOT / "experiments" / "results"

from sunnynet.baseline import solve_baseline_for_params  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.profiling import CacheCounter, PerfTracker  # noqa: E402


def forward_latency(seq_lens=(256, 512, 1024, 2048, 4096), threads=8,
                    repeats=3, vocab=8192):
    """实测两种架构在各序列长度下的单次前向耗时."""
    torch.set_num_threads(threads)
    cache = CacheCounter()
    out = {}
    for name, mk in (("sunnynet", "sunnynet"), ("transformer", "transformer")):
        rows = []
        for L in seq_lens:
            sunny = SunnyNet(SunnyConfig.small(vocab))
            if mk == "transformer":
                from sunnynet.baseline import TransformerLM
                cfg, _, _ = solve_baseline_for_params(sunny.num_params(True), vocab,
                                                      n_layers=4, max_len=L)
                model = TransformerLM(cfg)
            else:
                model = sunny
            model.eval()
            x = torch.randint(0, vocab, (1, L))
            with torch.no_grad():
                model(x)
                perf = PerfTracker()
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    model(x)
                    perf.update(L, time.perf_counter() - t0, None, cache)
            rows.append({"seq_len": L, "ms": round(perf.seconds / repeats * 1000, 3),
                         "ms_per_token": round(perf.ms_per_token, 4),
                         "params": model.num_params()})
        out[name] = rows
        print(f"  {name}: " + " ".join(f"T{r['seq_len']}={r['ms']:.1f}ms" for r in rows))
    return out


def train_at(arch: str, T: int, steps: int, threads: int, tag: str,
             lrs: dict) -> dict:
    out = RES / f"longseq_{arch}_T{T}.json"
    cmd = [PY, "-X", "utf8", str(ROOT / "train" / "train.py"),
           "--arch", arch, "--preset", "small", "--steps", str(steps),
           "--batch-size", "4", "--seq-len", str(T), "--lr", str(lrs[arch]),
           "--warmup", str(max(20, steps // 12)), "--threads", str(threads),
           "--val-every", str(max(50, steps // 4)), "--val-batches", "6",
           "--no-early-stop", "--out", str(out)]
    print(f"\n[{tag}] T={T} {arch} lr={lrs[arch]}")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return {"arch": arch, "T": T, "error": r.returncode}
    d = json.loads(out.read_text(encoding="utf-8"))
    s = d["summary"]
    print(f"  -> val={s['best_val_loss']} ppl={s['best_ppl']} "
          f"{s['ms_per_token_mean']:.4f} ms/tok rss={s['peak_rss_mb']:.0f}MB")
    return {"arch": arch, "T": T, "params": d["params"],
            "stack": d["params_non_embed"], "val": s["best_val_loss"],
            "ppl": s["best_ppl"], "ms_per_token": s["ms_per_token_mean"],
            "tokens_per_sec": s["tokens_per_sec_mean"],
            "peak_rss_mb": s["peak_rss_mb"], "steps": s["steps_done"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lens", default="256,512,1024")
    ap.add_argument("--only", default="", choices=["", "latency", "train"])
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    lrs = {"sunnynet": 0.003, "transformer": 0.006}
    report = {"lrs": lrs, "steps": args.steps, "threads": args.threads}

    if args.only != "train":
        print("== 前向延迟 vs 序列长度 ==")
        report["latency"] = forward_latency(threads=args.threads)
        (RES / "seq_scaling.json").write_text(
            json.dumps(report["latency"], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.only != "latency":
        print("== 长序列训练对照 ==")
        report["runs"] = []
        for T in [int(x) for x in args.lens.split(",")]:
            for arch in ("sunnynet", "transformer"):
                report["runs"].append(train_at(arch, T, args.steps, args.threads,
                                               "longseq", lrs))

    (RES / "longseq_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n序列长度  架构          参数栈      val      ppl    ms/tok  相对速度")
    for T in sorted({r["T"] for r in report.get("runs", []) if "error" not in r}):
        rows = {r["arch"]: r for r in report["runs"]
                if r.get("T") == T and "error" not in r}
        if len(rows) < 2:
            continue
        s, t = rows["sunnynet"], rows["transformer"]
        ratio = t["ms_per_token"] / s["ms_per_token"]
        for nm, r in (("sunnynet", s), ("transformer", t)):
            rel = f"{ratio:.3f}x" if nm == "sunnynet" else "1.000x"
            print(f"{T:8d}  {nm:12s} {r['stack']:>10,} {r['val']:8.4f} "
                  f"{r['ppl']:8.2f} {r['ms_per_token']:9.4f} {rel:>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())