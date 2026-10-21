"""数值锚定: GGUF 张量 vs 官方 .pth 的逐张量方向对照."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.runtime.gguf import GGUFModel  # noqa: E402

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
PTH = ROOT / "experiments/gguf/rwkv-x070-0.1b-v2.8.pth"


def main():
    g = GGUFModel(GGUF)
    w = torch.load(PTH, map_location="cpu", weights_only=True)
    w = {k: v.float().numpy() for k, v in w.items()}

    # GGUF 名 -> pth 名 (按层)
    L = g.get_int("rwkv7.block_count", 12)
    def map_name(n):
        if n == "token_embd.weight": return "emb.weight"
        if n == "output.weight": return "head.weight"
        if n == "token_embd_norm.weight": return "blocks.0.ln0.weight"
        if n == "token_embd_norm.bias": return "blocks.0.ln0.bias"
        if n == "output_norm.weight": return "ln_out.weight"
        if n == "output_norm.bias": return "ln_out.bias"
        if n.startswith("blk."):
            parts = n.split(".")
            i = parts[1]
            rest = ".".join(parts[2:])
            m = {
                "attn_norm.weight": "ln1.weight", "attn_norm.bias": "ln1.bias",
                "attn_norm_2.weight": "ln2.weight", "attn_norm_2.bias": "ln2.bias",
                "time_mix_w0.weight": "att.w0", "time_mix_w1.weight": "att.w1", "time_mix_w2.weight": "att.w2",
                "time_mix_a0.weight": "att.a0", "time_mix_a1.weight": "att.a1", "time_mix_a2.weight": "att.a2",
                "time_mix_v0.weight": "att.v0", "time_mix_v1.weight": "att.v1", "time_mix_v2.weight": "att.v2",
                "time_mix_g1.weight": "att.g1", "time_mix_g2.weight": "att.g2",
                "time_mix_k_k.weight": "att.k_k", "time_mix_k_a.weight": "att.k_a",
                "time_mix_r_k.weight": "att.r_k",
                "time_mix_key.weight": "att.key.weight",
                "time_mix_value.weight": "att.value.weight",
                "time_mix_receptance.weight": "att.receptance.weight",
                "time_mix_output.weight": "att.output.weight",
                "time_mix_ln.weight": "att.ln_x.weight", "time_mix_ln.bias": "att.ln_x.bias",
                "channel_mix_lerp_k.weight": "ffn.x_k",
                "channel_mix_key.weight": "ffn.key.weight",
                "channel_mix_value.weight": "ffn.value.weight",
            }
            if rest in m:
                return f"blocks.{i}.{m[rest]}"
        return None

    print(f"{'gguf 名':34s} {'np.shape':16s} {'pth.shape':16s} 方向   max|diff|")
    print("-" * 84)
    bad = 0
    checked = 0
    for n in g.names():
        p = map_name(n)
        if p is None or p not in w:
            continue
        a = g.tensor(n)
        b = w[p]
        checked += 1
        if a.shape == b.shape:
            d = np.abs(a - b).max()
            direction = "same"
        elif a.shape == b.shape[::-1]:
            d = np.abs(a - b.T).max()
            direction = "T"
        else:
            print(f"{n:34s} {str(a.shape):16s} {str(b.shape):16s} SHAPE MISMATCH")
            bad += 1
            continue
        flag = "" if d < 1e-5 else "  <-- 差异大"
        if d >= 1e-5:
            bad += 1
        print(f"{n:34s} {str(a.shape):16s} {str(b.shape):16s} {direction:5s}  {d:.3e}{flag}")
    print("-" * 84)
    print(f"checked {checked}, mismatch {bad}")


if __name__ == "__main__":
    main()