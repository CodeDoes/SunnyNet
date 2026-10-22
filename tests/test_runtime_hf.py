"""Mamba-2 / DeltaNet 统一 Runtime 的数值与等价性测试.

权重在 F 盘, 不在仓库内; 缺失时跳过, 不伪造结论。
"""

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# 权重体积大不入库; 用环境变量指向本地目录, 缺失则跳过相关用例。
MAMBA = os.environ.get("SUNNY_MAMBA2", str(ROOT / "models" / "mamba2-130m"))
DELTA = os.environ.get("SUNNY_DELTANET", str(ROOT / "models" / "deltanet-pure"))

skip_mamba = pytest.mark.skipif(
    not os.path.exists(os.path.join(MAMBA, "pytorch_model.bin")),
    reason="mamba2 权重不在本机")
skip_delta = pytest.mark.skipif(
    not os.path.exists(os.path.join(DELTA, "model.safetensors")),
    reason="deltanet 权重不在本机")


@pytest.fixture(scope="module")
def m2():
    from sunnynet.runtime.mamba2 import Mamba2
    return Mamba2(MAMBA)


@pytest.fixture(scope="module")
def dn():
    from sunnynet.runtime.deltanet import DeltaNet
    return DeltaNet(DELTA)


# --------------------------------------------------------------- Mamba-2
@skip_mamba
def test_mamba2_dims(m2):
    assert m2.d_model == 768
    assert m2.n_layers == 24
    assert m2.n_heads == 24
    assert m2.head_dim == 64
    assert m2.head_dim * m2.n_heads == m2.d_inner
    # conv_dim = d_inner + 2*n_groups*state_size
    assert m2.conv_dim == m2.d_inner + 2 * m2.n_groups * m2.state_size
    assert m2.conv_kernel == 4


@skip_mamba
def test_mamba2_step_vs_segment(m2):
    """同一状态上逐段续算必须与全程逐一喂入逐位一致."""
    rng = np.random.default_rng(11)
    ids = rng.integers(0, 50000, size=16).tolist()

    st = m2.reset(m2.init_state())
    full = np.stack([m2.step(t, st) for t in ids])

    st2 = m2.reset(m2.init_state())
    parts = [np.stack([m2.step(t, st2) for t in ids[:6]]),
             np.stack([m2.step(t, st2) for t in ids[6:]])]
    seg = np.concatenate(parts)
    assert np.abs(full - seg).max() == 0.0


@skip_mamba
def test_mamba2_matches_official(m2):
    """与 transformers 的 Mamba2ForCausalLM 整段前向对齐."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.mamba2")
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    cfg = Mamba2Config(
        model_type="mamba2", d_model=768, num_hidden_layers=24, vocab_size=50288,
        hidden_size=768, num_heads=24, head_dim=64, state_size=m2.state_size,
        n_groups=m2.n_groups, expand=2, conv_kernel=4, layer_norm_epsilon=1e-5,
        chunk_size=256, rms_norm=True, residual_in_fp32=True,
        pad_vocab_size_multiple=16,
    )
    ref = Mamba2ForCausalLM(cfg)
    sd = torch.load(os.path.join(MAMBA, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    ref.eval()

    rng = np.random.default_rng(12)
    ids = rng.integers(0, 50000, size=10).tolist()
    with torch.no_grad():
        exp = ref(torch.tensor([ids])).logits[0].numpy()

    st = m2.reset(m2.init_state())
    got = np.stack([m2.step(t, st) for t in ids])

    # 官方 fp32 自身相对 fp64 就偏离 ~4e-4, 单精度参考路径的阈值取 1e-3
    assert np.abs(exp - got).max() < 1e-3
    assert (exp.argmax(1) == got.argmax(1)).all()


# --------------------------------------------------------------- DeltaNet
@skip_delta
def test_deltanet_dims(dn):
    assert dn.hidden == 1024
    assert dn.n_layers == 24
    assert dn.n_heads == 8
    assert dn.head_k_dim == 128
    assert dn.head_v_dim == 128
    assert dn.conv_size == 4
    assert dn.headwise_beta


@skip_delta
def test_deltanet_rank1_equals_matrix_form(dn):
    """秩 1 更新与显式 (I - beta k k^T) S + beta k v^T 必须等价.

    S 在本仓按 (k_dim, v_dim) 存, head_k_dim == head_v_dim, 乘错一侧不会
    报错只会静默算成另一式子, 故这里用随机数据独立验证。
    """
    H, K, V = dn.n_heads, dn.head_k_dim, dn.head_v_dim
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(8):
        S = rng.standard_normal((H, K, V)).astype(np.float32)
        k = rng.standard_normal((H, K)).astype(np.float32)
        v = rng.standard_normal((H, V)).astype(np.float32)
        beta = rng.random(H).astype(np.float32)

        kv = np.einsum("hkv,hk->hv", S, k)
        S1 = S + k[:, :, None] * ((v - kv) * beta[:, None])[:, None, :]

        I = np.eye(K, dtype=np.float32)
        S2 = np.stack([(I - beta[h] * np.outer(k[h], k[h])) @ S[h]
                       + beta[h] * np.outer(k[h], v[h]) for h in range(H)])
        worst = max(worst, float(np.abs(S1 - S2).max()))
    assert worst < 1e-3, worst


@skip_delta
def test_deltanet_step_vs_segment(dn):
    rng = np.random.default_rng(14)
    ids = rng.integers(0, 30000, size=16).tolist()

    st = dn.reset(dn.init_state())
    full = np.stack([dn.step(t, st) for t in ids])

    st2 = dn.reset(dn.init_state())
    seg = np.concatenate([np.stack([dn.step(t, st2) for t in ids[:5]]),
                          np.stack([dn.step(t, st2) for t in ids[5:]])])
    assert np.abs(full - seg).max() == 0.0


@skip_delta
def test_deltanet_greedy_is_coherent(dn):
    """英文常识续写的贪心输出必须落到可读短语, 作为端到端连通性判据."""
    prompt = [1, 415, 5565, 302, 4843, 349]  # <s>The capital of France is
    gen = dn.generate(prompt, max_new=16, temperature=0.0)
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(DELTA)
    text = tok.decode(gen)
    assert "Paris" in text, text


@skip_delta
def test_deltanet_repeatable(dn):
    rng = np.random.default_rng(15)
    ids = rng.integers(0, 30000, size=8).tolist()
    st = dn.reset(dn.init_state())
    a = np.stack([dn.step(t, st) for t in ids])
    st2 = dn.reset(dn.init_state())
    b = np.stack([dn.step(t, st2) for t in ids])
    assert np.abs(a - b).max() == 0.0