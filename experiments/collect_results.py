"""汇总 experiments/results/*.json 为总表与 Markdown 片段.

只读已有真实结果文件, 不做任何估算或补值.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"


def row(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    s = d.get("summary") or {}
    a = d.get("args") or {}
    if not s:
        return None
    return {
        "file": path.stem,
        "arch": d.get("arch"),
        "params": d.get("params"),
        "stack": d.get("params_non_embed"),
        "ablation": a.get("ablation") or "-",
        "seed": a.get("seed"),
        "preset": a.get("preset"),
        "steps": s.get("steps_done"),
        "val": s.get("best_val_loss"),
        "ppl": s.get("best_ppl"),
        "train": s.get("final_train_loss"),
        "ms_tok": s.get("ms_per_token_mean"),
        "tok_s": s.get("tokens_per_sec_mean"),
        "rss": s.get("peak_rss_mb"),
        "gn": s.get("mean_grad_norm"),
        "cache": s.get("cache_hit_mean"),
        "cache_kind": s.get("cache_counter_kind"),
        "wall": s.get("wall_seconds"),
        "early": s.get("stopped_early"),
    }


def md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(t for _, t in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:,.4f}")
            elif isinstance(v, int) and abs(v) >= 1000:
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = [r for r in (row(p) for p in sorted(RES.glob("*.json"))) if r]
    if not rows:
        print("没有可用结果文件")
        return 1

    groups = {
        "Baseline comparison": lambda r: r["file"].startswith("baseline_"),
        "Ablation": lambda r: r["file"].startswith("ablation_"),
        "Seed stability": lambda r: r["file"].startswith("seed_"),
        "Scaling": lambda r: r["file"].startswith("scale_"),
        "Long-sequence training": lambda r: r["file"].startswith("longseq_"),
        "Learning-rate calibration": lambda r: r["file"].startswith("tune_"),
    }
    cols = [("file", "result file"), ("arch", "arch"), ("preset", "preset"),
            ("params", "total params"), ("stack", "stack params"), ("steps", "steps"),
            ("train", "train loss"), ("val", "val loss"), ("ppl", "ppl"),
            ("ms_tok", "ms/token"), ("tok_s", "token/s"), ("rss", "peak MB"),
            ("gn", "grad norm"), ("cache_kind", "memory probe")]

    out = []
    for name, pred in groups.items():
        sub = [r for r in rows if pred(r)]
        if not sub:
            continue
        print(f"\n### {name}\n")
        t = md_table(sub, cols)
        print(t)
        out.append(f"### {name}\n\n{t}\n")

    if args.out:
        Path(args.out).write_text("\n".join(out), encoding="utf-8")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())