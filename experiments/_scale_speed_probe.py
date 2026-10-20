"""单格: 构建指定 arch + preset 的模型, 测 batch=8 训练步耗时.

用法: python -X utf8 experiments/_scale_speed_probe.py <arch> <preset> <T>
每格独立子进程调用 (测量纪律).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import torch  # noqa: E402

from train import build_models  # noqa: E402


def main():
    arch, preset, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
    torch.set_num_threads(8)
    torch.manual_seed(0)
    V = 8192
    sunny, base, _ = build_models(arch, V, preset, None, T, 1234)
    model = sunny if arch == "sunnynet" else base
    model.train()
    x = torch.randint(0, V, (8, T))
    y = torch.randint(0, V, (8, T))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        lg = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    print(json.dumps({
        "arch": arch, "preset": preset, "T": T,
        "stack": model.num_params(non_embedding=True),
        "ms_per_tok": round(med * 1000 / (8 * T), 4),
        "iqr_ratio": round((ts[-1] - ts[0]) / med, 3),
    }))


if __name__ == "__main__":
    main()