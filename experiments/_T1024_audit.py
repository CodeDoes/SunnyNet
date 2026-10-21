"""T=1024 训练态速度与峰值内存审计: 七架构同协议.

动机: 质疑称 "T=1024 区间仍是速度最慢和峰值内存最高". 既有 on_scaling_t8.json
只测了 batch=1 纯前向, 且**从未记录过内存**. 本脚本补两个缺口:
  1) 训练态 (前向+反向+优化器步) 的 T=1024 ms/token;
  2) 训练态的激活内存开销, 用 current_rss(建完模型) 与 peak_rss(跑完一步) 之差度量,
     剔除 torch 运行时与参数量的固定开销.

测量纪律: 每个 (架构) 单元格单独起子进程, 进程内只构建一个模型 (堆碎片污染).
用法:
  python -X utf8 experiments/_T1024_audit.py [T] [batch] [archs]
  python -X utf8 experiments/_T1024_audit.py --_cell <arch> <T> <batch>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.baseline import TransformerLM, solve_baseline_for_params
from sunnynet.linear_baselines import BLOCKS, _LMBase, build_linear_baseline
from sunnynet.profiling import current_rss_mb, peak_rss_mb

arch, T, B = "{arch}", {T}, {B}
vocab = 8192
torch.set_num_threads(8); torch.manual_seed(1234)

ref = SunnyNet(SunnyConfig.small(vocab))
target = ref.num_params(non_embedding=True); del ref

if arch == "sunnynet":
    cfg = SunnyConfig.small(vocab); cfg.scan_chunk = 64
    model = SunnyNet(cfg)
elif arch == "transformer":
    cfg, _, _ = solve_baseline_for_params(target, vocab, n_layers=6,
                                          n_heads=8, max_len=max(T, 4096))
    model = TransformerLM(cfg)
else:
    cfg, _, _ = build_linear_baseline(arch, target, vocab, max_len=max(T, 4096), n_layers=4)
    model = _LMBase(cfg)
    for _ in range(cfg.n_layers):
        model.blocks.append(BLOCKS[arch](cfg))
    model._finish_init()

model.train()
stack = model.num_params(non_embedding=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, vocab, (B, T)); y = torch.randint(0, vocab, (B, T))

def step():
    lg = model(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); return float(loss.detach())

# peak_rss 是进程生命周期高水位: 必须在跑任何 step 之前取固定开销基线,
# 否则反向激活峰值会被算进基线, 差值失去意义.
rss_alloc = current_rss_mb()
for _ in range(2):
    step()
reps = 8
t0 = time.perf_counter()
lv = 0.0
for _ in range(reps):
    lv = step()
dt = (time.perf_counter() - t0) / reps
rss_peak = peak_rss_mb()

print("CELL " + json.dumps({{
    "arch": arch, "T": T, "batch": B, "stack": stack,
    "ms_per_token": round(dt * 1000 / (B * T), 5),
    "train_ms": round(dt * 1000, 2),
    "rss_alloc_mb": round(rss_alloc or 0, 1),
    "peak_rss_mb": round(rss_peak or 0, 1),
    "act_mb": round((rss_peak or 0) - (rss_alloc or 0), 1),
    "loss": round(lv, 4),
}}))
'''


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_cell":
        arch, T, B = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        code = CELL.format(root=str(ROOT).replace("\\", "\\\\"), arch=arch, T=T, B=B)
        r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                           text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("CELL ")]
        if not line:
            print(f"FAIL {arch}: {r.stderr[-300:]}", file=sys.stderr)
            sys.exit(1)
        print(line[0][5:])
        return

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    archs = sys.argv[3].split(",") if len(sys.argv) > 3 else [
        "sunnynet", "transformer", "linear_attn", "rwkv", "mamba", "hyena", "h3"]

    rows = []
    print(f"=== T={T} batch={B} 训练态, 每格独立子进程 ===")
    for a in archs:
        r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                            "--_cell", a, str(T), str(B)],
                           capture_output=True, text=True, encoding="utf-8")
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if not line:
            print(f"  {a:14s} FAILED: {r.stderr[-200:]}")
            continue
        d = json.loads(line[0])
        rows.append(d)
        print(f"  {a:14s} stack={d['stack']:>9,} {d['ms_per_token']:.5f} ms/tok "
              f"peak={d['peak_rss_mb']:.0f}MB act={d['act_mb']:.0f}MB")

    out = ROOT / "experiments" / "results" / f"_T{T}_audit.json"
    out.write_text(json.dumps({"T": T, "batch": B, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 排名 (T={T}, batch={B}) ===")
    sp = sorted(rows, key=lambda d: d["ms_per_token"])
    print("-- 速度 (ms/token, 越小越快) --")
    for i, d in enumerate(sp, 1):
        print(f"  {i}. {d['arch']:14s} {d['ms_per_token']:.5f}")
    mem = sorted(rows, key=lambda d: -d["peak_rss_mb"])
    print("-- 峰值内存 (大 → 小) --")
    for i, d in enumerate(mem, 1):
        print(f"  {i}. {d['arch']:14s} {d['peak_rss_mb']:.0f} MB "
              f"(激活 {d['act_mb']:.0f} MB)")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()