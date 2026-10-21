"""0.5B 在 6GB 卡上训练: 优化器状态 CPU offload.

问题: scale05b fp32 参数 1.9GB + AdamW 动量/方差 3.8GB + 梯度 1.9GB = 7.6GB > 6GB 显存.
前向可跑 (峰值 2.06GB), 训练步 OOM —— 卡在优化器状态, 不是架构.

解法: 参数与激活留 GPU, 优化器状态 (m, v) 常驻 CPU 内存, 更新时按张量搬运.
额外显存仅需一个参数张量的临时缓冲区, 优化器态 0 显存.

用法: python -X utf8 experiments/_05b_offload.py [T] [steps]
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

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
VOCAB = 8192
DEV = torch.device("cuda")

if not torch.cuda.is_available():
    print("RESULT " + json.dumps({"error": "CUDA 不可用"}))
    sys.exit(0)

torch.manual_seed(1234)
cfg = SunnyConfig.scale05b(VOCAB)
m = SunnyNet(cfg)
m = m.to(DEV)
torch.cuda.reset_peak_memory_stats()
print(f"build ok, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MB")

params = [p for p in m.parameters() if p.requires_grad]
# 优化器状态搬到 CPU, bf16 存储 (本机物理内存紧张, fp32 会触发页文件压力)
state = [{"m": torch.zeros_like(p, device="cpu", dtype=torch.bfloat16),
          "v": torch.zeros_like(p, device="cpu", dtype=torch.bfloat16)}
         for p in params]
n_state_gb = sum(s["m"].numel() + s["v"].numel() for s in state) * 2 / 2**30
print(f"优化器状态 {n_state_gb:.2f} GB 常驻 CPU (bf16)")

lr, b1, b2, eps = 3e-4, 0.9, 0.999, 1e-8
opt = torch.optim.SGD(params, lr=0.0)   # 占位, 手动更新
for g in opt.param_groups[0]["params"]:
    g.grad = None

x = torch.randint(0, VOCAB, (1, T), device=DEV)
y = torch.randint(0, VOCAB, (1, T), device=DEV)
lossf = torch.nn.CrossEntropyLoss()

t0 = time.time()
losses = []
for step in range(1, STEPS + 1):
    m.zero_grad(set_to_none=True)
    logits = m(x)
    loss = lossf(logits.view(-1, VOCAB), y.view(-1))
    loss.backward()
    losses.append(round(loss.item(), 4))

    # AdamW 手动步: 动量/方差用 bf16 存 CPU, 计算回 fp32
    with torch.no_grad():
        for p, s in zip(params, state):
            g = p.grad
            gm = g.detach().to("cpu", dtype=torch.float32, copy=True)
            s["m"] = (s["m"].float().mul_(b1).add_(gm, alpha=1 - b1)).bfloat16()
            s["v"] = (s["v"].float().mul_(b2).addcmul_(gm, gm, value=1 - b2)).bfloat16()
            mh = s["m"].float() / (1 - b1 ** step)
            vh = s["v"].float() / (1 - b2 ** step)
            upd = (mh / (vh.sqrt() + eps)).to(DEV, non_blocking=True)
            p.add_(upd, alpha=-lr)
            p.add_(p, alpha=-lr * 0.01)   # weight decay
            del gm, mh, vh, upd
    if step == 1 or step % 5 == 0:
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"  step {step:3d} loss={losses[-1]:.4f} peak_gpu={peak:.0f}MB "
              f"elapsed={time.time()-t0:.1f}s")

out = {
    "T": T, "steps": STEPS, "params": sum(p.numel() for p in params),
    "opt_state_gb": round(n_state_gb, 2),
    "losses": losses,
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
    "elapsed_s": round(time.time() - t0, 1),
    "loss_delta": round(losses[-1] - losses[0], 4),
}
print("RESULT " + json.dumps(out))
(ROOT / "experiments" / "results" / "_05b_offload.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")