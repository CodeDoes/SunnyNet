"""实验编排: 基线对照 / 消融 / 种子稳定性 / 缩放趋势.

所有实验共用同一份 token 流、同一 batch/seq 配置、同一 token 预算与同一线程数,
保证对照公平. 每个实验的结果落 experiments/results/*.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "train" / "train.py"
RESULTS = ROOT / "experiments" / "results"


def run_one(tag: str, extra: list[str], log_name: str) -> dict:
    out = RESULTS / f"{log_name}.json"
    cmd = [PY, "-X", "utf8", str(TRAIN), "--out", str(out)] + extra
    print(f"\n{'='*78}\n[{tag}] {' '.join(extra)}\n{'='*78}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[{tag}] 训练进程返回码 {r.returncode}")
        return {"tag": tag, "error": f"returncode={r.returncode}"}
    return json.loads(out.read_text(encoding="utf-8"))


def summarize(log: dict) -> dict:
    s = log.get("summary", {})
    return {
        "arch": log.get("arch"),
        "params": log.get("params"),
        "params_non_embed": log.get("params_non_embed"),
        "steps_done": s.get("steps_done"),
        "best_val_loss": s.get("best_val_loss"),
        "best_ppl": s.get("best_ppl"),
        "final_train_loss": s.get("final_train_loss"),
        "ms_per_token": s.get("ms_per_token_mean"),
        "tokens_per_sec": s.get("tokens_per_sec_mean"),
        "peak_rss_mb": s.get("peak_rss_mb"),
        "mean_grad_norm": s.get("mean_grad_norm"),
        "cache_hit_mean": s.get("cache_hit_mean"),
        "wall_seconds": s.get("wall_seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["baseline", "ablation", "seeds", "scale", "probe"])
    ap.add_argument("--preset", default="small")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", default="1234,2024,3407")
    ap.add_argument("--mix", default="")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    common = ["--preset", args.preset, "--steps", str(args.steps),
              "--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len),
              "--lr", str(args.lr), "--threads", str(args.threads),
              "--val-every", str(max(50, args.steps // 8))]
    if args.mix:
        common += ["--mix", args.mix]

    report: dict = {"exp": args.exp, "common": common, "runs": []}

    if args.exp == "probe":
        r = run_one("speed-probe", common + ["--arch", "sunnynet", "--seed", "1234"],
                    "probe_sunnynet")
        report["runs"].append({"tag": "sunnynet", "summary": summarize(r)})
        r = run_one("speed-probe", common + ["--arch", "transformer", "--seed", "1234"],
                    "probe_transformer")
        report["runs"].append({"tag": "transformer", "summary": summarize(r)})

    elif args.exp == "baseline":
        for arch in ("sunnynet", "transformer"):
            r = run_one(f"baseline/{arch}", common + ["--arch", arch, "--seed", "1234"],
                        f"baseline_{arch}")
            report["runs"].append({"tag": arch, "summary": summarize(r),
                                   "history": r.get("history", [])})

    elif args.exp == "ablation":
        variants = {
            "full": [],
            "no_decay_gate": ["--ablation", "decay_gate=0"],
            "no_circulant": ["--ablation", "circulant=0"],
            "no_local_taps": ["--ablation", "local_taps=0"],
            "no_current": ["--ablation", "current=0"],
            "none": ["--ablation", "decay_gate=0,circulant=0,local_taps=0,current=0"],
        }
        for name, ex in variants.items():
            r = run_one(f"ablation/{name}", common + ["--arch", "sunnynet",
                        "--seed", "1234"] + ex, f"ablation_{name}")
            report["runs"].append({"tag": name, "summary": summarize(r),
                                   "history": r.get("history", [])})

    elif args.exp == "seeds":
        for sd in [int(s) for s in args.seeds.split(",")]:
            r = run_one(f"seed/{sd}", common + ["--arch", "sunnynet", "--seed", str(sd)],
                        f"seed_{sd}")
            report["runs"].append({"tag": f"seed{sd}", "seed": sd,
                                   "summary": summarize(r)})

    elif args.exp == "scale":
        for preset in ("small", "proto20m"):
            for arch in ("sunnynet", "transformer"):
                r = run_one(f"scale/{preset}/{arch}",
                            common + ["--preset", preset, "--arch", arch, "--seed", "1234"],
                            f"scale_{preset}_{arch}")
                report["runs"].append({"tag": f"{preset}/{arch}",
                                       "summary": summarize(r)})

    p = RESULTS / f"report_{args.exp}.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {p}")

    print("\n摘要")
    hdr = f"{'run':22s} {'params':>10s} {'val':>8s} {'ppl':>8s} {'ms/tok':>9s} {'tok/s':>9s} {'RSS MB':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in report["runs"]:
        s = r["summary"]
        if "error" in r:
            print(f"{r['tag']:22s} ERROR {r['error']}")
            continue
        p_ = f"{s['params']:,}" if s.get("params") else "-"
        print(f"{r['tag']:22s} {p_:>10s} {str(s.get('best_val_loss')):>8s} "
              f"{str(s.get('best_ppl')):>8s} {str(s.get('ms_per_token')):>9s} "
              f"{str(s.get('tokens_per_sec')):>9s} {str(s.get('peak_rss_mb')):>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())