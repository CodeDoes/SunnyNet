"""RWKV-7 C 内核 (runtime/rwkv7.c) 的自检.

权重是真实外部文件, 不随仓库分发; 缺失时整模块 skip。
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
BLOB = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"


def _have():
    try:
        import gguf  # noqa: F401
    except ImportError:
        return False
    return GGUF.exists() and BLOB.exists()


pytestmark = pytest.mark.skipif(not _have(), reason="缺少 gguf 包或 RWKV-7 权重/blob")

PROMPT = "The Eiffel Tower is located in the city of"


def test_c_kernel_matches_numpy_reference():
    """C 整模型单步与 numpy 参考逐 token logits 一致."""
    import sys
    sys.path.insert(0, str(ROOT))
    from sunnynet.runtime.rwkv7 import RWKV7
    from sunnynet.runtime.rwkv7_c import SnRWKV7

    ref = RWKV7(GGUF)
    c = SnRWKV7(BLOB, ref.vocab)
    ids = ref.tok.encode(PROMPT)[:8]

    rs = ref.reset(ref.init_state())
    c.reset()
    worst = 0.0
    same = 0
    for t in ids:
        rl = ref.step(t, rs)
        cl = np.array(c.step(t), dtype=np.float32)
        worst = max(worst, float(np.abs(rl - cl).max()))
        same += int(np.argmax(rl) == np.argmax(cl))
    assert worst < 1e-3, f"max|diff| {worst:.3e} 超出 bf16 精度"
    assert same == len(ids)


def test_c_kernel_prefill_matches_sequential():
    """prefill 与逐步 step 的结果一致."""
    import sys
    sys.path.insert(0, str(ROOT))
    from sunnynet.runtime.rwkv7 import RWKV7
    from sunnynet.runtime.rwkv7_c import SnRWKV7

    ref = RWKV7(GGUF)
    ids = ref.tok.encode(PROMPT)

    a = SnRWKV7(BLOB, ref.vocab)
    a.reset()
    last = a.prefill(ids)

    b = SnRWKV7(BLOB, ref.vocab)
    b.reset()
    for t in ids:
        last2 = b.step(t)
    assert float(np.abs(np.array(last) - np.array(last2)).max()) == 0.0


def test_c_kernel_generates_coherent_text():
    """C 内核贪心生成的基本连贯性 (非语义断言, 只查不退化)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from sunnynet.runtime.rwkv7 import RWKV7
    from sunnynet.runtime.rwkv7_c import SnRWKV7

    ref = RWKV7(GGUF)
    c = SnRWKV7(BLOB, ref.vocab)
    c.reset()
    ids = ref.tok.encode(PROMPT)
    logits = np.array(c.prefill(ids), dtype=np.float32)
    out = list(ids)
    for _ in range(16):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        logits = np.array(c.step(nxt), dtype=np.float32)
    text = ref.tok.decode(out)
    assert "Paris" in text, text