"""GGUF 只读加载 + RWKV-7 单步递推的自检.

权重是真实外部文件 (BlinkDL/rwkv-7-world 经 llama.cpp 转换的 GGUF), 不随仓库分发.
文件缺失时整模块 skip.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
PTH = ROOT / "experiments/gguf/rwkv-x070-0.1b-v2.8.pth"


def _have():
    try:
        import gguf  # noqa: F401
    except ImportError:
        return False
    return GGUF.exists()


pytestmark = pytest.mark.skipif(not _have(), reason="缺少 gguf 包或 RWKV-7 权重文件")


def test_gguf_meta_and_tokens():
    from sunnynet.runtime.gguf import GGUFModel
    g = GGUFModel(GGUF)
    assert g.get_int("rwkv7.block_count") == 12
    assert g.get_int("rwkv7.embedding_length") == 768
    assert g.get_int("rwkv7.wkv.head_size") == 64
    toks = g.tokens()
    assert len(toks) == 65536
    assert toks[1] == b"\x00"
    # 词表字节反转义后应能找到 utf-8 编码的 '中'
    assert any(t == "\u4e2d".encode("utf-8") for t in toks)


def test_gguf_tensor_matches_pth_bitwise():
    from sunnynet.runtime.gguf import GGUFModel
    g = GGUFModel(GGUF)
    w = torch.load(PTH, map_location="cpu", weights_only=True)
    a = g.tensor("token_embd.weight")
    b = w["emb.weight"].float().numpy()
    assert a.shape == b.shape
    assert float(np.abs(a - b).max()) == 0.0


def test_rwkv7_step_matches_pth_reference():
    """GGUF 驱动实现与官方 .pth 独立参考实现逐 token 数值等价."""
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments._rwkv7_gguf_equiv import Ref
    from sunnynet.runtime.rwkv7 import RWKV7

    mine = RWKV7(GGUF)
    ref = Ref(PTH)
    ids = mine.tok.encode("The Eiffel Tower is located in the city of")[:8]

    rs, ms = ref.init(), mine.reset(mine.init_state())
    worst = 0.0
    same_argmax = 0
    for t in ids:
        rl = ref.step(t, rs)
        ml = mine.step(t, ms)
        worst = max(worst, float(np.abs(rl - ml).max()))
        same_argmax += int(np.argmax(rl) == np.argmax(ml))
    assert worst < 1e-4, f"max|diff| {worst:.3e} 超出 bf16 精度"
    assert same_argmax == len(ids)


def test_rwkv7_tokenizer_roundtrip():
    from sunnynet.runtime.rwkv7 import RWKV7
    m = RWKV7(GGUF)
    for s in ["Hello world", "\u4e2d\u56fd\u7684\u9996\u90fd\u662f"]:
        assert m.tok.decode(m.tok.encode(s)) == s