"""本机已有权重上的 C 全模型解码一致性 (附加测试, 无权重时跳过)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_ood/sunnynet.pt"
BLOB = ROOT / "experiments/ckpts_ood/sunnynet.runtime.bin"


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_runtime_full_model_matches_torch_decode():
    """C 整模型逐 token 前向与 torch forward_step 数值一致, argmax 全同."""
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel
    torch.set_num_threads(1)
    m, cfg, _ = load_model(str(CKPT))
    m.eval()
    rt = SnModel(BLOB, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                 cfg.layers, 8, int(cfg.tie_embeddings))
    torch.manual_seed(7)
    toks = torch.randint(0, cfg.vocab_size, (24,)).tolist()
    s = m.init_state(1)
    with torch.no_grad():
        for t in toks:
            lg_t, s = m.forward_step(torch.tensor([t]), s)
            lg_c = torch.tensor(rt.step(t))
            assert float((lg_t[0] - lg_c).abs().max()) < 1e-3
            assert int(lg_t[0].argmax()) == int(lg_c.argmax())


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_runtime_prefill_equals_stepwise():
    """C 侧 prefill(n) 与逐步 step 的末 token logits 一致."""
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel
    torch.set_num_threads(1)
    _, cfg, _ = load_model(str(CKPT))
    torch.manual_seed(3)
    toks = torch.randint(0, cfg.vocab_size, (16,)).tolist()

    a = SnModel(BLOB, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                cfg.layers, 8, int(cfg.tie_embeddings))
    lg_pref = torch.tensor(a.prefill(toks))
    b = SnModel(BLOB, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                cfg.layers, 8, int(cfg.tie_embeddings))
    last = None
    for t in toks:
        last = torch.tensor(b.step(t))
    assert float((lg_pref - last).abs().max()) < 1e-4