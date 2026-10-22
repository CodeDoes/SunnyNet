"""Mamba-2 numpy/C 运行时与 transformers 官方实现的数值对照.

用增量解码 (use_cache=True) 对齐: 官方每步只喂 1 个 token, 与本仓单步递推同口径。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))


def make_cfg(n_groups, state_size):
    return {
        "model_type": "mamba2",
        "d_model": 768,
        "n_layer": 24,
        "vocab_size": 50288,
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 16,
        "tie_embeddings": True,
        "ssm_cfg": {"layer": "Mamba2"},
        "hidden_size": 768,
        "num_hidden_layers": 24,
        "num_heads": 24,
        "head_dim": 64,
        "state_size": state_size,
        "n_groups": n_groups,
        "expand": 2,
        "conv_kernel": 4,
        "layer_norm_epsilon": 1e-5,
        "chunk_size": 256,
        "rescale_prenorm_residual": False,
    }


def official_logits(ids, cfg, incremental=True):
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config

    c = Mamba2Config(**cfg)
    m = Mamba2ForCausalLM(c)
    sd = torch.load(MDIR + r"\pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print("  [warn] missing:", missing[:6], "..." if len(missing) > 6 else "")
    if unexpected:
        print("  [warn] unexpected:", unexpected[:6], "..." if len(unexpected) > 6 else "")
    m.eval()

    if not incremental:
        with torch.no_grad():
            out = m(torch.tensor([ids])).logits[0]
        return out.numpy(), None

    outs = []
    cache = None
    with torch.no_grad():
        for i, t in enumerate(ids):
            out = m(torch.tensor([[t]]), use_cache=True,
                    cache_params=cache, is_first_iteration=(i == 0))
            cache = out.cache_params
            outs.append(out.logits[0, -1].numpy())
    return np.stack(outs), None


def main():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 50277, size=8).tolist()

    for n_groups, state_size in [(8, 16), (1, 128), (2, 64), (4, 32), (24, 64 // 24)]:
        if n_groups * state_size != 128:
            continue
        cfg = make_cfg(n_groups, state_size)
        print(f"--- n_groups={n_groups} state_size={state_size} ---")
        try:
            ref, _ = official_logits(ids, cfg, incremental=True)
        except Exception as e:
            print("  official failed:", type(e).__name__, e)
            continue
        from sunnynet.runtime.mamba2 import Mamba2
        try:
            m = Mamba2(MDIR, n_groups=n_groups, state_size=state_size)
        except Exception as e:
            print("  runtime failed:", type(e).__name__, e)
            continue
        st = m.reset(m.init_state())
        mine = np.stack([m.step(t, st) for t in ids])
        d = np.abs(ref - mine)
        print(f"  ref shape {ref.shape} mine {mine.shape}")
        print(f"  max|diff| = {d.max():.3e}  mean = {d.mean():.3e}")
        print(f"  argmax ref   {ref[-1].argmax()}  mine {mine[-1].argmax()}")


if __name__ == "__main__":
    main()