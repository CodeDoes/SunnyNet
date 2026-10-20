"""0.5B (scale05b) 在 6GB GPU 上的可行性阶梯测试.

逐级递进, 每级独立捕获 OOM, 记录峰值显存:
  L0 构建(meta/cpu) -> L1 构建到 cuda fp32 -> L2 前向 B=1 T=512
  -> L3 前向 B=1 T=1024 -> L4 SGD 训练步 -> L5 AdamW 训练步
不追求跑满训练, 只回答"能到哪一级".

用法: python -X utf8 experiments/_05b_gpu_ladder.py [T]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
VOCAB = 8192
out = {"T": T, "torch": torch.__version__,
       "cuda": torch.cuda.is_available()}
if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"error": "CUDA 不可用"}))
    sys.exit(0)

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
total_mb = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
out["gpu"] = name
out["gpu_total_mb"] = round(total_mb)
print(f"GPU={name} total={total_mb:.0f}MB")


def peak():
    return round(torch.cuda.max_memory_allocated() / 2 ** 20)


def fresh():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


cfg = SunnyConfig.scale05b(VOCAB)

# --- L0: meta 计数 (不占显存) ---
try:
    with torch.device("meta"):
        mm = SunnyNet(cfg)
    out["params_total"] = mm.num_params()
    out["params_stack"] = mm.num_params(non_embedding=True)
    print(f"L0 meta: total={out['params_total']:,} stack={out['params_stack']:,}")
except Exception as e:
    out["L0_error"] = str(e)[:200]

# --- L1: 构建到 cuda fp32 ---
fresh()
m = None
try:
    t0 = time.perf_counter()
    m = SunnyNet(cfg).to(dev).eval()
    torch.cuda.synchronize()
    out["L1_build_s"] = round(time.perf_counter() - t0, 2)
    out["L1_peak_mb"] = peak()
    print(f"L1 build cuda: {out['L1_build_s']}s peak={out['L1_peak_mb']}MB")
except RuntimeError as e:
    out["L1_error"] = str(e)[:200]
    print(f"L1 FAIL: {out['L1_error']}")
    print("RESULT " + json.dumps(out, ensure_ascii=False))
    sys.exit(0)

# --- L2/L3: 前向 ---
x = None
for label, tt in (("L2", 512), ("L3", 1024)):
    try:
        fresh()
        x = torch.randint(0, VOCAB, (1, tt), device=dev)
        with torch.no_grad():
            y = m(x)
            torch.cuda.synchronize()
        out[f"{label}_fwd_T{tt}_peak_mb"] = peak()
        out[f"{label}_logits"] = list(y.shape)
        print(f"{label} fwd T={tt}: peak={out[f'{label}_fwd_T{tt}_peak_mb']}MB")
        del y, x
        x = None
    except RuntimeError as e:
        out[f"{label}_error"] = str(e)[:200]
        print(f"{label} FAIL T={tt}: {out[f'{label}_error'][:80]}")

# --- L4: SGD 训练步 ---
try:
    fresh()
    m.train()
    x = torch.randint(0, VOCAB, (1, T), device=dev)
    yt = torch.randint(0, VOCAB, (1, T), device=dev)
    opt = torch.optim.SGD(m.parameters(), lr=1e-4)
    lg = m(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), yt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    out["L4_sgd_peak_mb"] = peak()
    out["L4_loss"] = round(float(loss.detach()), 4)
    print(f"L4 SGD step: peak={out['L4_sgd_peak_mb']}MB loss={out['L4_loss']}")
    del opt, lg, loss, x, yt
except RuntimeError as e:
    out["L4_error"] = str(e)[:200]
    print(f"L4 FAIL: {out['L4_error'][:100]}")

# --- L5: AdamW 训练步 (2x 状态) ---
try:
    fresh()
    m.train()
    x = torch.randint(0, VOCAB, (1, T), device=dev)
    yt = torch.randint(0, VOCAB, (1, T), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lg = m(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), yt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    out["L5_adamw_peak_mb"] = peak()
    print(f"L5 AdamW step: peak={out['L5_adamw_peak_mb']}MB")
except RuntimeError as e:
    out["L5_error"] = str(e)[:200]
    print(f"L5 FAIL: {out['L5_error'][:100]}")

out["gpu_free_mb"] = round(torch.cuda.mem_get_info()[0] / 2 ** 20)
print("RESULT " + json.dumps(out, ensure_ascii=False))