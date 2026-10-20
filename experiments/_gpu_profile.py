"""GPU 上核心扫描 / 竞争门控的真实耗时占比剖析.

CPU 侧只能用"替换为极简实现"估上界; GPU 侧可直接按 CUDA kernel 计时.
回答: 在 GPU 上把这两段融合, 收益空间比 CPU 上大还是小.

用法: python -X utf8 experiments/_gpu_profile.py [T] [B]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
VOCAB = 8192

if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"error": "CUDA 不可用"}))
    sys.exit(0)

dev = torch.device("cuda")
print(f"GPU={torch.cuda.get_device_name(0)} T={T} B={B}")

cfg = SunnyConfig.small(VOCAB)
cfg.scan_chunk = 64
torch.manual_seed(0)
m = SunnyNet(cfg).to(dev).eval()

x = torch.randint(0, VOCAB, (B, T), device=dev)


def run_profile(mode: str, n: int = 12):
    for _ in range(3):
        with torch.set_grad_enabled(mode == "train"):
            y = m(x)
            y.sum().backward() if mode == "train" else None
        if mode == "train":
            m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            with torch.set_grad_enabled(mode == "train"):
                y = m(x)
                if mode == "train":
                    y.sum().backward()
                    m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    return prof


def aggregate(prof, mode: str):
    """按 aten op 名聚合 self CUDA 时间, 并归到 scan/compete/其他."""
    rows = defaultdict(lambda: [0.0, 0])
    for ev in prof.key_averages():
        key = ev.key
        cuda = getattr(ev, "self_device_time_total", 0.0) or 0.0
        if cuda <= 0 and getattr(ev, "self_cuda_time_total", 0.0):
            cuda = ev.self_cuda_time_total
        if cuda <= 0:
            continue
        rows[key][0] += cuda
        rows[key][1] += ev.count
    total = sum(v[0] for v in rows.values()) or 1.0
    out = {
        "total_us": round(total),
        "top": sorted(
            [{"op": k, "us": round(v[0]), "pct": round(100 * v[0] / total, 2), "n": v[1]}
             for k, v in rows.items()],
            key=lambda r: -r["us"])[:14],
    }
    # 归组: cumsum 属扫描; sigmoid 相关属竞争门控 (粗分, 仅作比例参考)
    scan_like = sum(v[0] for k, v in rows.items()
                    if "cumsum" in k or "logspace" in k or "scan" in k)
    gate_like = sum(v[0] for k, v in rows.items()
                    if k.count("sigmoid") or "compete" in k)
    out["scan_like_pct"] = round(100 * scan_like / total, 2)
    out["gate_like_pct"] = round(100 * gate_like / total, 2)
    return out


res = {"gpu": torch.cuda.get_device_name(0), "T": T, "B": B, "torch": torch.__version__}
for mode in ("eval", "train"):
    prof = run_profile(mode)
    res[mode] = aggregate(prof, mode)
    print(f"--- {mode} total={res[mode]['total_us']}us "
          f"scan_like={res[mode]['scan_like_pct']}% gate_like={res[mode]['gate_like_pct']}%")
    for r in res[mode]["top"][:8]:
        print(f"    {r['op'][:46]:48s} {r['us']:7d}us {r['pct']:5.1f}% x{r['n']}")

print("RESULT " + json.dumps(res, ensure_ascii=False))