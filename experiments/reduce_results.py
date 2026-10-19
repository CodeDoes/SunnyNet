"""把分散的训练结果聚合为 make_figures 期望的 report_*.json.

字段只搬运, 不重算不补值; 缺失字段写 None.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"

GROUPS = [
    ("report_baseline.json", [("sunnynet", "baseline_sunnynet"),
                              ("transformer", "baseline_transformer")]),
    ("report_ablation.json", [("full", "ablation_full"),
                              ("no_decay_gate", "ablation_no_decay_gate"),
                              ("no_circulant", "ablation_no_circulant"),
                              ("no_local_taps", "ablation_no_local_taps"),
                              ("no_current", "ablation_no_current"),
                              ("none", "ablation_none")]),
    ("report_seeds.json", [("seed1234", "seed_1234"),
                           ("seed2024", "seed_2024"),
                           ("seed3407", "seed_3407")]),
    ("report_scale.json", [("proto20m/sunnynet", "scale_proto20m_sunnynet"),
                           ("proto20m/transformer", "scale_proto20m_transformer")]),
]


def _summary(d: dict) -> dict:
    s = d.get("summary") or {}
    a = d.get("args") or {}
    return {
        "arch": d.get("arch"),
        "preset": a.get("preset"),
        "params": d.get("params"),
        "params_non_embed": d.get("params_non_embed"),
        "steps_done": s.get("steps_done"),
        "best_val_loss": s.get("best_val_loss"),
        "best_ppl": s.get("best_ppl"),
        "final_train_loss": s.get("final_train_loss"),
        "ms_per_token": s.get("ms_per_token_mean"),
        "tokens_per_sec": s.get("tokens_per_sec_mean"),
        "peak_rss_mb": s.get("peak_rss_mb"),
        "mean_grad_norm": s.get("mean_grad_norm"),
        "cache_hit_mean": s.get("cache_hit_mean"),
        "cache_counter_kind": s.get("cache_counter_kind"),
        "wall_seconds": s.get("wall_seconds"),
    }


def _run(tag: str, path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"tag": tag, "summary": _summary(d), "history": d.get("history", [])}


def main() -> int:
    for out_name, pairs in GROUPS:
        runs = []
        for tag, stem in pairs:
            p = RES / f"{stem}.json"
            if p.exists():
                runs.append(_run(tag, p))
        if not runs:
            continue
        out = RES / out_name
        out.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"  {out_name}: {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())