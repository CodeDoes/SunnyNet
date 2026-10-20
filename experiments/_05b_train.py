"""0.5B (scale05b) 在 6GB GPU 上的多步训练探针.

只回答"能否用 SGD 让 loss 下降", 不做收敛判断.
AdamW 因双动量状态超显存不可用, 故固定 SGD; 每步独立记录 loss / 峰值显存 / 耗时.
真实语料 (data/proc), 不使用随机 token.

用法: python -X utf8 experiments/_05b_train.py [steps] [T] [lr]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from build_corpus import load_tokens  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
T = int(sys.argv[2]) if len(sys.argv) > 2 else 128
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
VOCAB = 8192

out = {"torch": torch.__version__, "steps": STEPS, "T": T, "lr": LR,
       "cuda": torch.cuda.is_available()}
if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"error": "CUDA 不可用"}))
    sys.exit(0)

dev = torch.device("cuda")
out["gpu"] = torch.cuda.get_device_name(0)
out["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
print(f"GPU={out['gpu']} total={out['gpu_total_mb']}MB  steps={STEPS} T={T} lr={LR}")


def load_domain():
    return {d: load_tokens(d, "train").astype(np.int64)
            for d in ("zh_ancient", "en_literature", "code_python")}


arrs = load_domain()
doms = list(arrs)
rng = np.random.default_rng(1234)


def sample_batch():
    d = doms[int(rng.integers(0, len(doms)))]
    a = arrs[d]
    s = int(rng.integers(0, a.size - T - 1))
    w = a[s:s + T + 1]
    return (torch.from_numpy(w[:-1].copy()).unsqueeze(0).to(dev),
            torch.from_numpy(w[1:].copy()).unsqueeze(0).to(dev))


cfg = SunnyConfig.scale05b(VOCAB)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
m = SunnyNet(cfg).to(dev).train()
torch.cuda.synchronize()
out["build_s"] = round(time.perf_counter() - t0, 2)
out["params_total"] = m.num_params()
print(f"build {out['build_s']}s params={out['params_total']:,}")

opt = torch.optim.SGD(m.parameters(), lr=LR)
warmup = max(5, STEPS // 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warmup))
clip = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
curve = []
ok = True
for step in range(STEPS):
    x, y = sample_batch()
    lg = m(x)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), clip))
    opt.step()
    sched.step()
    torch.cuda.synchronize()
    lv = float(loss.detach())
    curve.append({"step": step + 1, "loss": round(lv, 4), "gn": round(gn, 3),
                  "lr": round(opt.param_groups[0]["lr"], 6)})
    if step < 3 or (step + 1) % 10 == 0 or step == STEPS - 1:
        print(f"  step {step+1:3d} loss={lv:.4f} gn={gn:.3f} "
              f"lr={opt.param_groups[0]['lr']:.2e} "
              f"peak={torch.cuda.max_memory_allocated()/2**20:.0f}MB")
    del x, y, lg, loss
    if not math.isfinite(lv) or not math.isfinite(gn):
        out["diverged_at"] = step + 1
        ok = False
        break

out["curve"] = curve
out["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
if len(curve) >= 2:
    out["loss_first"] = curve[0]["loss"]
    out["loss_last"] = curve[-1]["loss"]
    out["loss_delta"] = round(curve[-1]["loss"] - curve[0]["loss"], 4)
out["ok"] = ok
out["note"] = "SGD 短程探针, 无收敛判断; AdamW 显存不可用"
print("RESULT " + json.dumps(out, ensure_ascii=False))