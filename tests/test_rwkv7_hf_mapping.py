"""FLA 命名权重 -> C blob 的映射与端到端一致性.

需要权重: SUNNY_RWKV7_HF (FLA safetensors 目录)。不设则整文件跳过。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sunnynet.runtime.gguf import GGUFModel
from sunnynet.runtime.vocab import RWKVVocab

ROOT = Path(__file__).resolve().parents[1]
HF_DIR = os.environ.get("SUNNY_RWKV7_HF")
GGUF_191M = ROOT / "experiments" / "gguf" / "rwkv7-191M-world-bf16.gguf"
GGUF_Q4 = ROOT / "experiments" / "gguf" / "rwkv7-191M-world-bf16.q4.bin"
VOCAB = ROOT / "experiments" / "gguf" / "rwkv_vocab_v20230424.txt"

pytestmark = pytest.mark.skipif(
    not HF_DIR or not Path(HF_DIR).is_dir(),
    reason="需设置 SUNNY_RWKV7_HF 指向 FLA safetensors 目录")


@pytest.fixture(scope="module")
def sd():
    from safetensors.torch import load_file
    return load_file(str(Path(HF_DIR) / "model.safetensors"), device="cpu")


def test_vocab_file_matches_gguf_tokens():
    """官方词表文件与 GGUF 内嵌词表应逐 token 相同 (文件 id 从 1 起)。"""
    if not VOCAB.exists() or not GGUF_191M.exists():
        pytest.skip("缺词表或 GGUF")
    v = RWKVVocab(str(VOCAB))
    g = GGUFModel(str(GGUF_191M)).tokens()
    assert len(v.idx2tok) >= 65000
    for i in (1, 2, 100, 6699, 33155, 65529):
        assert v.idx2tok[i] == g[i], f"token {i} 不一致"


def test_vocab_encode_matches_gguf():
    if not VOCAB.exists() or not GGUF_191M.exists():
        pytest.skip("缺词表或 GGUF")
    from sunnynet.runtime.rwkv7 import _Tok
    v = RWKVVocab(str(VOCAB))
    g = _Tok(GGUFModel(str(GGUF_191M)).tokens())
    for s in ("The Eiffel Tower is located in the city of", "你好，世界",
              "Hello world!", "def fibonacci(n):"):
        assert v.encode(s) == g.encode(s), s


def test_tensor_mapping_bitwise_identical(sd):
    """FLA 张量 -> GGUF 张量: 前 6 组同名张量应逐位相同。"""
    if not GGUF_191M.exists():
        pytest.skip("缺 191M GGUF")
    g = GGUFModel(str(GGUF_191M))
    L = 1
    pairs = [
        (f"model.layers.{L}.attn.r_proj.weight", f"blk.{L}.time_mix_receptance.weight"),
        (f"model.layers.{L}.attn.k_proj.weight", f"blk.{L}.time_mix_key.weight"),
        (f"model.layers.{L}.attn.v_proj.weight", f"blk.{L}.time_mix_value.weight"),
        (f"model.layers.{L}.attn.o_proj.weight", f"blk.{L}.time_mix_output.weight"),
        (f"model.layers.{L}.attn.k_k", f"blk.{L}.time_mix_k_k"),
        (f"model.layers.{L}.attn.k_a", f"blk.{L}.time_mix_k_a"),
    ]
    for a, b in pairs:
        x = sd[a].float().numpy()
        y = g.tensor(b).astype(np.float32)
        assert x.shape == y.shape, a
        assert np.abs(x - y).max() == 0.0, a


def test_lerp_fused_splits_into_x_vectors(sd):
    """time_mix_lerp_fused 的 6 段应等于 x_r/x_w/x_k/x_v/x_a/x_g。"""
    if not GGUF_191M.exists():
        pytest.skip("缺 191M GGUF")
    g = GGUFModel(str(GGUF_191M))
    L = 1
    lerp = g.tensor(f"blk.{L}.time_mix_lerp_fused.weight").reshape(6, -1)
    for i, nm in enumerate(["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]):
        x = sd[f"model.layers.{L}.attn.{nm}"].float().numpy().reshape(-1)
        assert np.abs(lerp[i] - x).max() == 0.0, nm


def test_lora_bias_mapping(sd):
    """w0/a0 分别是 w_lora/a_lora 第二层的 bias。"""
    if not GGUF_191M.exists():
        pytest.skip("缺 191M GGUF")
    g = GGUFModel(str(GGUF_191M))
    L = 1
    assert np.abs(g.tensor(f"blk.{L}.time_mix_w0.weight")
                  - sd[f"model.layers.{L}.attn.w_lora.lora.2.bias"].float().numpy()).max() == 0.0
    assert np.abs(g.tensor(f"blk.{L}.time_mix_a0.weight")
                  - sd[f"model.layers.{L}.attn.a_lora.lora.2.bias"].float().numpy()).max() == 0.0


def test_pre_norm_maps_to_token_embd_norm(sd):
    """第 0 层 pre_norm 等于 GGUF 的 token_embd_norm。"""
    if not GGUF_191M.exists():
        pytest.skip("缺 191M GGUF")
    g = GGUFModel(str(GGUF_191M))
    for suf in ("weight", "bias"):
        a = sd[f"model.layers.0.pre_norm.{suf}"].float().numpy()
        b = g.tensor(f"token_embd_norm.{suf}").astype(np.float32)
        assert np.abs(a - b).max() == 0.0, suf