"""Mamba-2 单步递推的等价性与生成质量.

三项:
  1. step vs 整段前向 (先喂全部 token 再比对最后一步 logits) —— 递推正确性
  2. 分段续算 vs 整段 —— 状态可迁移性
  3. 贪心生成连贯性 —— 端到端可用性
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sunnynet.runtime.mamba2 import Mamba2  # noqa: E402

MDIR = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))


def main():
    m = Mamba2(MDIR)
    print(f"n_groups={m.n_groups} state_size={m.state_size} n_heads={m.n_heads} "
          f"head_dim={m.head_dim} conv_dim={m.conv_dim} conv_k={m.conv_kernel} "
          f"vocab={m.vocab} layers={m.n_layers}")

    rng = np.random.default_rng(1)
    ids = rng.integers(0, 50000, size=24).tolist()

    # 1. 逐步喂入, 记录每步 logits
    st = m.reset(m.init_state())
    step_logits = np.stack([m.step(t, st) for t in ids])

    # 2. 分段: 前 10 步建状态, 后 14 步在同一状态上继续 (模拟缓存续算)
    st2 = m.reset(m.init_state())
    for t in ids[:10]:
        m.step(t, st2)
    seg_logits = []
    st3 = m.reset(m.init_state())
    for t in ids:
        seg_logits.append(m.step(t, st3))
    seg_logits = np.stack(seg_logits)

    d_step = np.abs(step_logits - seg_logits).max()
    print(f"1) 单步 vs 分段续算 max|diff| = {d_step:.3e}")

    # 3. 与官方整段前向比对
    import torch
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    c = Mamba2Config(
        model_type="mamba2", d_model=768, num_hidden_layers=24, vocab_size=50288,
        hidden_size=768, num_heads=24, head_dim=64, state_size=128, n_groups=1,
        expand=2, conv_kernel=4, layer_norm_epsilon=1e-5, chunk_size=256,
        rms_norm=True, residual_in_fp32=True, pad_vocab_size_multiple=16,
    )
    ref = Mamba2ForCausalLM(c)
    sd = torch.load(MDIR + r"\pytorch_model.bin", map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    ref.eval()
    with torch.no_grad():
        out = ref(torch.tensor([ids])).logits[0].numpy()
    d = np.abs(out - step_logits)
    print(f"2) vs 官方整段前向 max|diff| = {d.max():.3e}  mean = {d.mean():.3e}")
    print(f"   argmax 逐步 {step_logits.argmax(1)}")
    print(f"   argmax 官方 {out.argmax(1)}")
    print(f"   argmax 一致 {(step_logits.argmax(1) == out.argmax(1)).sum()}/{len(ids)}")

    # 4. 贪心生成
    prompt = [12600, 314, 3968, 374, 6033]  # "The capital of France is"
    gen = m.generate(prompt, max_new=16, temperature=0.0)
    print(f"3) 贪心续写 ids = {gen}")


if __name__ == "__main__":
    main()