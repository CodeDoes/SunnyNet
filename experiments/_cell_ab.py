"""A/B 单格子进程入口: 训练一个变体并报告 best ppl."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.screen_variants import variant_model  # noqa: E402
from sunnynet.tokenizer import BPETokenizer  # noqa: E402
from train.train import DEFAULT_MIX, DomainMixer, evaluate, lr_at  # noqa: E402

variant, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 8)

tk = BPETokenizer.load(ROOT / "data" / "proc" / "tokenizer.json")
V = tk.vocab_size
m = variant_model(variant, V, seed)
mx = DomainMixer(DEFAULT_MIX, 256, 8, "train", seed=seed)
vm = DomainMixer(DEFAULT_MIX, 256, 8, "val", seed=1)
vb = vm.val_batches(8)

opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
best = float("inf")
for s in range(steps):
    lr = lr_at(s, steps, 3e-3, 60)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = mx.batch()
    lg = m(x)
    loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if (s + 1) % 150 == 0 or s == steps - 1:
        best = min(best, evaluate(m, vb))

print("RESULT " + json.dumps({"variant": variant, "seed": seed, "steps": steps,
                              "best_val": round(best, 6),
                              "ppl": round(math.exp(min(best, 20)), 4)}))