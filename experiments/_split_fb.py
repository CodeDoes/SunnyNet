"""训练步耗时分解: 前向 / 反向 / 优化器, 并按算子族归并.

回答"训练速度差距在哪一段". 每格独立子进程.

用法: python -X utf8 experiments/_split_fb.py <arch> [T] [B] [reps]
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.baseline import solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import _make, build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

ARCH = sys.argv[1] if len(sys.argv) > 1 else "sunnynet"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
VOCAB = 8192

torch.set_num_threads(8)
torch.manual_seed(1234)


def build():
    if ARCH == "sunnynet":
        cfg = SunnyConfig.small(VOCAB)
        cfg.scan_chunk = 64
        return SunnyNet(cfg)
    ref = SunnyNet(SunnyConfig.small(VOCAB))
    target = ref.num_params(non_embedding=True)
    del ref
    cfg, _, _ = build_linear_baseline(ARCH, target, VOCAB, max_len=max(T, 2048),
                                      n_layers=6)
    return _make(ARCH, cfg)


m = build().train()
opt = torch.optim.SGD(m.parameters(), lr=1e-4)
x = torch.randint(0, VOCAB, (B, T))
y = torch.randint(0, VOCAB, (B, T))


def fwd():
    return torch.nn.functional.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))


def train_step():
    loss = fwd()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def timeit(fn, n):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


t_fwd = timeit(fwd, REPS)
t_train = timeit(train_step, REPS)

# --- 算子族归并 (纯测量, 看哪族最重) ---
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(REPS):
        train_step()

FAM = [
    ("mm/addmm", r"aten::(mm|addmm|bmm)"),
    ("log_softmax", r"log_softmax"),
    ("mul", r"aten::mul_?\b"),
    ("add", r"aten::add_?\b"),
    ("cat", r"aten::cat"),
    ("copy", r"aten::copy_"),
    ("fill/zeros", r"aten::(fill_|zeros|zero_)"),
    ("sigmoid", r"sigmoid"),
    ("cumsum", r"aten::cumsum"),
    ("exp/log", r"aten::(exp|log)\b"),
    ("conv", r"convolution|mkldnn_conv"),
    ("roll", r"aten::roll"),
    ("layer_norm", r"layer_norm"),
    ("softwedge等", r"aten::(abs|div|neg|sub)\b"),
]
fam_ms = defaultdict(float)
fam_calls = defaultdict(int)
for ev in prof.key_averages():
    cpu = getattr(ev, "self_cpu_time_total", 0.0) or 0.0
    if cpu <= 0:
        continue
    key = ev.key
    hit = None
    for name, pat in FAM:
        if re.search(pat, key):
            hit = name
            break
    hit = hit or "其他"
    fam_ms[hit] += cpu / 1000
    fam_calls[hit] += ev.count

total = sum(fam_ms.values()) or 1.0
rows = sorted([{"fam": k, "ms": round(v, 2), "pct": round(100 * v / total, 2),
                "calls": fam_calls[k]} for k, v in fam_ms.items()],
              key=lambda r: -r["ms"])

out = {"arch": ARCH, "T": T, "B": B,
       "fwd_ms": round(t_fwd, 2), "train_ms": round(t_train, 2),
       "bwd_opt_ms": round(t_train - t_fwd, 2),
       "bwd_pct": round(100 * (t_train - t_fwd) / t_train, 1),
       "fwd_pct": round(100 * t_fwd / t_train, 1),
       "fam_total_ms": round(total, 2), "fam": rows}
print(f"[{ARCH}] T={T} B={B}  fwd={out['fwd_ms']}ms ({out['fwd_pct']}%)  "
      f"bwd+opt={out['bwd_opt_ms']}ms ({out['bwd_pct']}%)  train={out['train_ms']}ms")
for r in rows[:12]:
    print(f"  {r['fam']:14s} {r['ms']:8.2f}ms {r['pct']:5.1f}% calls={r['calls']}")
print("RESULT " + json.dumps(out, ensure_ascii=False))