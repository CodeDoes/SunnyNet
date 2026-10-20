"""读 experiments/results/sc*_*.json, 汇总为规模-架构对照表.

只读真实结果, 缺格显示 "—", 不做任何估算.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"

ARCHS = ["sunnynet", "mamba", "h3", "hyena", "rwkv", "linear_attn", "transformer"]


def load(tag: str) -> dict[str, dict]:
    out = {}
    for a in ARCHS:
        p = RES / f"sc_{tag}_{a}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d.get("summary") or {}
        if not s:
            continue
        out[a] = {
            "params": d.get("params"),
            "stack": d.get("params_non_embed"),
            "ppl": s.get("best_ppl"),
            "ms_tok": s.get("ms_per_token_mean"),
            "rss": s.get("peak_rss_mb"),
            "steps": s.get("steps_done"),
        }
    return out


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}" if isinstance(v, int) and abs(v) >= 1000 else str(v)


def table(rows: dict[str, dict], col: str, header: str, nd=2) -> str:
    lines = [f"| 架构 | {header} |", "|---|---|"]
    for a in ARCHS:
        if a in rows:
            lines.append(f"| `{a}` | {fmt(rows[a].get(col), nd)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="如 small20m 100m")
    args = ap.parse_args()
    data = {t: load(t) for t in args.tags}

    for t, rows in data.items():
        if not rows:
            print(f"### {t}: 无结果文件\n")
            continue
        print(f"### {t}")
        sn = rows.get("sunnynet")
        print(f"\n参数量(栈): " + ", ".join(
            f"{a}={fmt(rows[a]['stack'])}" for a in rows))
        if sn:
            print("\n**ppl (越低越好)**\n")
            print(table(rows, "ppl", "ppl"))
            print("\n**ms/token (越低越好)**\n")
            print(table(rows, "ms_tok", "ms/token", 4))
        print()

    # 交叉汇总
    if len(data) > 1:
        print("### 跨规模趋势 (sunnynet)\n")
        print("| 规模 | ppl | ms/token | 栈参数 | 峰值MB |")
        print("|---|---|---|---|---|")
        for t, rows in data.items():
            sn = rows.get("sunnynet")
            if sn:
                print(f"| {t} | {fmt(sn['ppl'])} | {fmt(sn['ms_tok'], 4)} | "
                      f"{fmt(sn['stack'])} | {fmt(sn['rss'], 1)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())