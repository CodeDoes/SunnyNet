"""核验 small 规模下 sunnynet 与六架构的参数量对齐 (meta device 计数).

对照公平性纪律: 参数量必须用实例化计数, 禁止手推公式.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.baseline import solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import LINEAR_ARCHS, _make, build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

VOCAB = 8192
cfg = SunnyConfig.small(VOCAB)
s = SunnyNet(cfg)
sunny_total = s.num_params()
sunny_stack = s.num_params(non_embedding=True)
print(f"sunnynet: total={sunny_total:,} stack={sunny_stack:,} dim={cfg.dim} L={cfg.layers}")
print(f"          注意 sunnynet 总参数含 embedding {VOCAB}*{cfg.dim}={VOCAB*cfg.dim:,}")

rows = []
for arch in ("transformer", *LINEAR_ARCHS):
    if arch == "transformer":
        bcfg, n, rel = solve_baseline_for_params(sunny_stack, VOCAB, n_layers=6,
                                                 n_heads=8, max_len=2048)
        print(f"{arch:14s} n_layers={bcfg.n_layers} d={bcfg.d_model} stack≈{n:,}")
        rows.append({"arch": arch, "stack": n})
        continue
    lcfg, n, rel = build_linear_baseline(arch, sunny_stack, VOCAB, max_len=2048,
                                         n_layers=6)
    m = _make(arch, lcfg)
    tot = m.num_params()
    stk = m.num_params(non_embedding=True)
    rows.append({"arch": arch, "d_model": lcfg.d_model, "n_layers": lcfg.n_layers,
                 "stack": stk, "total": tot,
                 "stack_bias_pct": round(100 * (stk - sunny_stack) / sunny_stack, 3)})
    print(f"{arch:14s} d={lcfg.d_model:4d} L={lcfg.n_layers} "
          f"stack={stk:,} (偏差 {rows[-1]['stack_bias_pct']:+.2f}%) total={tot:,}")

print()
print(f"结论: sunnynet stack={sunny_stack:,}, 同行 stack 均应对齐到此值")
print("RESULT " + json.dumps({"sunnynet": {"total": sunny_total, "stack": sunny_stack},
                              "baselines": rows}, ensure_ascii=False))