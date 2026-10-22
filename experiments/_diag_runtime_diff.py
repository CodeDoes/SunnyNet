"""定位 max|diff| 来源: 逐 token 打印 torch vs C 的差值轨迹。

若首 token 就有量级误差 => 转置/布局 bug; 若从 1e-7 缓慢增长
=> 递推对微小舍入的放大 (需与改动前的实现对比确认)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"


def main():
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel

    torch.set_num_threads(1)
    m, cfg, _ = load_model(str(CKPT))
    m.eval()
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    V = d["vocab_size"]
    torch.manual_seed(7)
    toks = torch.randint(0, V, (20,)).tolist()

    h = SnModel(BLOB, V, d["dim"], d["inner"], d["rank"], d.get("taps", 3),
                d["layers"], d.get("group", 8), 1)
    h.reset()

    st = m.init_state(1)
    print(" tok   max|diff|   torch_argmax  C_argmax  torch_logit_sum")
    with torch.no_grad():
        for i, t in enumerate(toks):
            lt, st = m.forward_step(torch.tensor([t]), st)
            lc = np.asarray(h.step(t), dtype=np.float32)
            lt_np = lt[0].float().numpy()
            df = float(np.abs(lt_np - lc).max())
            print(f"{i:4d}  {df:10.3e}   {int(lt_np.argmax()):11d}  "
                  f"{int(lc.argmax()):8d}  {float(lt_np.sum()):12.4f}")
            if df > 1e-2:
                print("  -> 首现量级误差, 检查该步之前的权重布局")
                break


if __name__ == "__main__":
    main()