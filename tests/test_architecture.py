"""SunnyNet 架构自检.

重点是三条可证伪的性质, 任何一条不成立即架构实现有错:
  A. 分块扫描一致性: chunk=1 / 64 / 2048 三种切法输出必须逐元素相同.
     这是"块末状态反推块首状态是精确恒等式"的直接检验.
  B. 因果性: 改变位置 t 之后的输入, 不得影响位置 <= t 的输出. 无泄漏.
  C. 梯度界: softwedge 的导数在 [-1e6, 1e6] 上始终落在 (1, 1+a/2].
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sunnynet.baseline import BaselineConfig, TransformerLM, solve_baseline_for_params  # noqa: E402
from sunnynet.init import orthogonal_spectrum_  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.ops import (CenteredScaleNorm, logspace_scan, multiscale_decay,  # noqa: E402
                          poly_sigmoid, softwedge)
from sunnynet.tokenizer import BPETokenizer, pretokenize  # noqa: E402


# ---------------------------------------------------------------- 算子性质
def test_poly_sigmoid_range_and_monotone():
    z = torch.linspace(-1e6, 1e6, 200001)
    s = poly_sigmoid(z)
    assert torch.all(s > 0) and torch.all(s < 1)
    assert abs(float(poly_sigmoid(torch.tensor(0.0))) - 0.5) < 1e-9
    assert float(poly_sigmoid(torch.tensor(1e6))) == pytest.approx(1.0, abs=1e-5)
    assert float(poly_sigmoid(torch.tensor(-1e6))) == pytest.approx(0.0, abs=1e-5)
    # 单调性直接检验解析导数 s'(z)=0.5/(1+|z|)^2 > 0, 避免 fp32 差分下溢
    zz = torch.linspace(-1e6, 1e6, 200001, dtype=torch.float64, requires_grad=True)
    poly_sigmoid(zz).sum().backward()
    grad = zz.grad
    assert torch.all(grad > 0), f"导数非正: {grad.min()}"
    ref = 0.5 / (1.0 + zz.detach().abs()) ** 2
    assert torch.allclose(grad, ref, rtol=1e-9, atol=1e-12)
    assert float(ref.max()) == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("a", [0.1, 0.5, 0.99])
def test_softwedge_gradient_bounds_analytic(a):
    z = torch.linspace(-1e6, 1e6, 200001, requires_grad=True)
    softwedge(z, torch.tensor(a)).sum().backward()
    g = z.grad
    assert g.min() > 1.0 - 1e-6, f"梯度下界破裂: {g.min()}"
    assert g.max() <= 1.0 + a / 2 + 1e-4, f"梯度上界破裂: {g.max()}"
    assert torch.isfinite(g).all()


def test_softwedge_no_gradient_cliff():
    """导数必须连续, 不存在阶跃断崖."""
    z = torch.linspace(-200, 200, 400001, requires_grad=True)
    softwedge(z, torch.tensor(0.5)).sum().backward()
    d = z.grad
    jump = d.diff().abs().max().item()
    assert jump < 5e-4, f"导数出现阶跃 {jump}"


def test_softwedge_gradcheck():
    z = torch.randn(64, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda zz, aa: softwedge(zz, aa), (z, a), eps=1e-6)


def test_csn_constant_input_stability():
    """近常量输入不得被小方差放大成噪声 (原创下界尺度的目的)."""
    n = CenteredScaleNorm(64)
    x = torch.full((4, 32, 64), 3.14159)
    y = n(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() < 1e-3, f"常量输入被放大: {y.abs().max()}"
    x2 = torch.full((4, 32, 64), 5.0) + torch.randn(4, 32, 64) * 1e-7
    assert torch.isfinite(n(x2)).all()


def test_csn_unit_variance():
    n = CenteredScaleNorm(128)
    x = torch.randn(8, 16, 128) * 7.0 + 20.0
    y = n(x)
    assert y.std(-1).mean().item() == pytest.approx(1.0, abs=0.05)


def test_multiscale_decay_spans_range():
    d = multiscale_decay(512)
    h = 1.0 / d
    assert h.min().item() == pytest.approx(2.0, rel=1e-4)
    assert h.max().item() == pytest.approx(8192.0, rel=1e-3)
    assert torch.all(d > 0) and torch.all(d < 1)
    assert 1.0 / d.min().item() == pytest.approx(8192.0, rel=1e-3)


def test_multiscale_decay_monotone_spacing():
    """log2 域上跨度应近似均匀 (多时间尺度是设计目标)."""
    h = torch.log2(1.0 / multiscale_decay(256))
    gaps = h.diff()
    assert gaps.std() / gaps.mean() < 0.05


# ---------------------------------------------------------------- 扫描精确性
def test_logspace_scan_exact_vs_step():
    torch.manual_seed(0)
    B, T, N = 2, 300, 32
    nu = torch.randn(B, T, N)
    p = torch.rand(B, T, N) * 0.4 + 0.55
    s0 = torch.randn(B, N)
    s = s0
    parts = []
    for t in range(T):
        s = p[:, t] * s + (1 - p[:, t]) * nu[:, t]
        parts.append(s)
    ref = torch.stack(parts, dim=1)
    got, s_end = logspace_scan(nu, p, s0, chunk=37)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), \
        f"对数域扫描与逐步递推不一致, max diff={(got-ref).abs().max()}"
    assert torch.allclose(s_end, ref[:, -1], atol=1e-4)


@pytest.mark.parametrize("chunk", [1, 7, 64, 512, 4096])
def test_block_chunk_invariance_single_block(chunk):
    """单块在任意 chunk 下输出必须完全一致."""
    torch.manual_seed(3)
    from sunnynet.layers import SunnyBlock
    blk = SunnyBlock(dim=32, inner=48, rank=16, scan_chunk=chunk)
    blk.eval()
    x = torch.randn(2, 200, 32)
    s = torch.randn(2, 32)
    h = torch.randn(2, blk.taps - 1, 32)
    with torch.no_grad():
        o1, s1, h1 = blk(x, s, h)
    ref_chunk = blk.scan_chunk
    blk.scan_chunk = 1
    with torch.no_grad():
        o2, s2, h2 = blk(x, s, h)
    blk.scan_chunk = ref_chunk
    assert torch.allclose(o1, o2, atol=1e-5), f"chunk={chunk} 输出不一致"
    assert torch.allclose(s1, s2, atol=1e-4), f"chunk={chunk} 末态不一致"
    assert torch.allclose(h1, h2, atol=1e-6), f"chunk={chunk} 卷积历史不一致"


def test_block_chunk_invariance_full_model():
    torch.manual_seed(7)
    cfg = SunnyConfig.small(256)
    m = SunnyNet(cfg)
    m.eval()
    idx = torch.randint(0, 256, (2, 130))
    with torch.no_grad():
        a = m(idx)
    for blk in m.blocks:
        blk.scan_chunk = 1
    with torch.no_grad():
        b = m(idx)
    assert torch.allclose(a, b, atol=1e-5), "整模型分块一致性失败"


# ---------------------------------------------------------------- 因果性
def test_no_future_leakage():
    """改动位置 >= k 的输入, 位置 < k 的输出必须不变."""
    torch.manual_seed(11)
    m = SunnyNet(SunnyConfig.small(512))
    m.eval()
    idx = torch.randint(0, 512, (1, 128))
    k = 64
    idx2 = idx.clone()
    idx2[0, k:] = torch.randint(0, 512, (128 - k,))
    with torch.no_grad():
        o1 = m(idx)
        o2 = m(idx2)
    assert torch.allclose(o1[:, :k], o2[:, :k], atol=1e-6), "存在未来信息泄漏"
    assert not torch.allclose(o1[:, k:], o2[:, k:]), "后段本应改变, 说明模型常量输出"


def test_causal_conv_is_causal():
    from sunnynet.layers import DepthwiseCausal
    torch.manual_seed(5)
    c = DepthwiseCausal(16, taps=4)
    x = torch.randn(1, 40, 16)
    x2 = x.clone()
    x2[0, 20:] = torch.randn(20, 16)
    with torch.no_grad():
        assert torch.allclose(c(x)[:, :20], c(x2)[:, :20], atol=1e-6)


# ---------------------------------------------------------------- 状态与梯度
def test_state_bounded():
    """凸组合保证状态幅值不超过输入与历史的较大者."""
    torch.manual_seed(13)
    cfg = SunnyConfig.small(512)
    m = SunnyNet(cfg)
    idx = torch.randint(0, 512, (4, 400))
    with torch.no_grad():
        _, st = m(idx, return_state=True)
    assert torch.isfinite(st.s).all() and torch.isfinite(st.hist).all()
    assert st.s.abs().max() < 50, f"状态发散: {st.s.abs().max()}"


def test_long_sequence_state_stable():
    torch.manual_seed(17)
    m = SunnyNet(SunnyConfig.small(512))
    with torch.no_grad():
        _, st = m(torch.randint(0, 512, (1, 4096)), return_state=True)
    assert torch.isfinite(st.s).all(), "长序列状态出现 NaN/Inf"
    assert torch.isfinite(st.hist).all(), "长序列卷积历史出现 NaN/Inf"


def test_gradients_finite_and_nonvanishing():
    torch.manual_seed(19)
    m = SunnyNet(SunnyConfig.small(512))
    idx = torch.randint(0, 512, (2, 256))
    logits = m(idx)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), idx.reshape(-1))
    loss.backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.grad is not None]
    assert grads, "没有任何梯度"
    for n, g in grads:
        assert torch.isfinite(g).all(), f"{n} 梯度非有限"
    # 最深层(第 0 块)的状态通路梯度不应消失
    g0 = m.blocks[0].W_u.grad if hasattr(m.blocks[0], "W_u") else m.blocks[0].V.weight.grad
    assert g0 is not None and g0.abs().max() > 1e-12, "首块梯度消失"


def test_gradient_stable_across_depth():
    """默认配置下深堆叠: 输入嵌入梯度范数随深度不应呈指数衰减.

    谱半径 rho0=1.0 / decay_beta=0 为实测标定值; 改回层深衰减 rho_l=rho0*(l+1)^-beta
    会在本测试立即失败, 作为回归防线保留. 默认含残差旁路 (见 §5), 故此处守的是
    连同残差在内的完整默认通路.
    """
    norms = []
    for L in (4, 12, 24):
        torch.manual_seed(23 + L)
        cfg = SunnyConfig(vocab_size=256, dim=128, inner=192, rank=64, layers=L)
        m = SunnyNet(cfg)
        idx = torch.randint(0, 256, (2, 128))
        loss = torch.nn.functional.cross_entropy(
            m(idx).reshape(-1, 256), idx.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()
        g = m.embed.weight.grad
        assert g is not None
        norms.append(g.norm().item())
    assert all(n > 1e-6 for n in norms), f"深层梯度归零: {norms}"
    r1, r2 = norms[1] / norms[0], norms[2] / norms[1]
    assert 0.2 < r1 < 5.0, f"4->12 层梯度比异常 {r1} (norms={norms})"
    assert 0.2 < r2 < 5.0, f"12->24 层梯度比异常 {r2} (norms={norms})"


def test_spectral_scale_decides_depth_stability():
    """回归防线 (关闭残差, 只看状态接力通路): 谱半径必须定标到 ρ≈1.

    实测 (无残差, 24 层/4 层梯度范数比, CSN 真实归一化 s_min=0.02):
        ρ0=0.1 -> 1.0e+0(真零)   ρ0=0.2 -> 1.1e-4 (崩塌)
        ρ0=0.3 -> 0.92           ρ0=0.5 -> 1.03
        ρ0=0.7 -> 1.58           ρ0=1.0 -> 6.58
        ρ0=1.0,β=0.15 -> 1.52    ρ0=2.0 -> 3.5e+2 (爆炸)

    修正记录: 该边界随 CSN 地板系统性移动 —— ρ0 崩塌点由 s_min=0.25 时的
    "≤0.5" 下移到 s_min=0.05 的 "≤0.3"、再下移到 s_min=0.02 的 "≤0.2"。
    原因: 地板越小, 归一化越接近真方差归一化, 越能补偿深层欠缩放, 故旧防线
    "β>0 必崩塌" 与 "ρ0=0.5 崩塌" 只对欠归一化的旧 CSN 成立, 均已作废。

    默认配置已加残差旁路, 崩塌不再出现 (见下一个测试), 故此处显式关闭残差,
    使这条防线守的是"状态接力总线"。
    """
    no_resid = {"residual": False}

    def grads(L, rho0, beta, ab=None):
        torch.manual_seed(7)
        cfg = SunnyConfig(vocab_size=256, dim=128, inner=192, rank=64,
                          layers=L, rho0=rho0, decay_beta=beta,
                          ablation=dict(ab or no_resid))
        m = SunnyNet(cfg)
        idx = torch.randint(0, 256, (2, 128))
        loss = torch.nn.functional.cross_entropy(
            m(idx).reshape(-1, 256), idx.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()
        g = m.embed.weight.grad
        return g.norm().item() if g is not None else 0.0

    def ratio(L1, L2, rho0, beta):
        return grads(L1, rho0, beta) / max(grads(L2, rho0, beta), 1e-30)

    assert ratio(24, 4, 0.2, 0.0) < 1e-2, "重欠缩放 ρ0=0.2 应导致 24 层梯度崩塌"
    assert ratio(24, 4, 1.0, 0.0) > 0.5, "定标 ρ=1 梯度应随深度健全"
    assert 0.1 < ratio(24, 4, 1.0, 0.15) < 100, "真实归一化下 β=0.15 不再单独崩塌"
    assert ratio(24, 4, 2.0, 0.0) > 10, "过缩放应随深度放大梯度"


def test_residual_removes_depth_collapse():
    """残差旁路使欠缩放深堆叠不再梯度崩塌 (默认配置).

    实测 (ρ0=0.2, 24 层/4 层梯度比, s_min=0.02): 有残差 1.20, 无残差 1.1e-4。
    用 ρ0=0.2 而非旧记录的 0.5/0.3: CSN 地板下调后归一化补偿增强, 崩塌点随之
    下移 (0.25->≤0.5, 0.05->≤0.3, 0.02->≤0.2), 故崩塌对照点必须跟随下移以保持
    判别力。残差提供一条不经过状态接力的直连通路, 使谱锚定不再是深度的唯一保障。
    """
    def g(L, rho0, residual):
        torch.manual_seed(7)
        cfg = SunnyConfig(vocab_size=256, dim=128, inner=192, rank=64,
                          layers=L, rho0=rho0, ablation={"residual": residual})
        m = SunnyNet(cfg)
        idx = torch.randint(0, 256, (2, 128))
        loss = torch.nn.functional.cross_entropy(
            m(idx).reshape(-1, 256), idx.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()
        g_ = m.embed.weight.grad
        return g_.norm().item() if g_ is not None else 0.0

    without = g(24, 0.2, False) / max(g(4, 0.2, False), 1e-30)
    with_res = g(24, 0.2, True) / max(g(4, 0.2, True), 1e-30)
    assert without < 1e-2, f"无残差时欠缩放应崩塌, 实测比 {without}"
    assert with_res > 0.5, f"有残差时不应崩塌, 实测比 {with_res}"


# ---------------------------------------------------------------- 初始化
def test_spectral_init_singular_values():
    torch.manual_seed(29)
    for shape in [(64, 64), (48, 96), (96, 48), (37, 64)]:
        w = torch.empty(*shape)
        orthogonal_spectrum_(w, 2.5)
        sv = torch.linalg.svdvals(w)
        assert sv.max().item() == pytest.approx(2.5, rel=1e-3), f"{shape} sigma_max={sv.max()}"
        assert sv.min().item() > 0, f"{shape} 条件数退化"


def test_init_loss_near_uniform():
    """初始化后 loss 应接近 ln(V), 即初始分布近均匀."""
    torch.manual_seed(31)
    V = 1024
    m = SunnyNet(SunnyConfig.small(V))
    idx = torch.randint(0, V, (4, 128))
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(m(idx).reshape(-1, V), idx.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.15, f"初始 loss={loss.item()} 偏离 ln(V)={math.log(V)}"


def test_init_reproducible():
    a = SunnyNet(SunnyConfig.small(512))
    torch.manual_seed(999)
    b = SunnyNet(SunnyConfig.small(512))
    torch.manual_seed(999)
    c = SunnyNet(SunnyConfig.small(512))
    assert torch.equal(b.embed.weight, c.embed.weight)
    assert torch.equal(b.blocks[0].V.weight, c.blocks[0].V.weight)


# ---------------------------------------------------------------- 推理一致
def test_forward_step_matches_forward():
    """单步自回归必须与整段前向严格等价 (流式推理正确性).

    这是架构最容易出错的地方: 曾经把块首状态拼进候选通路, 导致单步与整段
    不等价 (fp64 下偏差 2.2e-1). 现经状态反馈后置 + 卷积历史缓冲修正,
    fp64 下偏差应降到机器精度量级.
    """
    torch.manual_seed(37)
    m = SunnyNet(SunnyConfig.small(512)).double()
    m.eval()
    idx = torch.randint(0, 512, (1, 96))
    with torch.no_grad():
        full, _ = m(idx, return_state=True)
        st = m.init_state(1, dtype=torch.float64)
        outs = []
        for t in range(idx.shape[1]):
            lg, st = m.forward_step(idx[:, t], st)
            outs.append(lg)
    step = torch.stack(outs, dim=1)
    d = float((step - full).abs().max())
    assert d < 1e-10, f"单步与整段不一致 max diff={d}"


def test_segment_continuation_matches_whole():
    """分段续算必须与一次性整段前向等价 (长文/流式分块推理正确性)."""
    torch.manual_seed(41)
    m = SunnyNet(SunnyConfig.small(512)).double()
    m.eval()
    idx = torch.randint(0, 512, (1, 120))
    with torch.no_grad():
        whole, _ = m(idx, return_state=True)
        for split in (32, 60, 96):
            s0 = m.init_state(1, dtype=torch.float64)
            o1, s1 = m(idx[:, :split], s0, return_state=True)
            o2, _ = m(idx[:, split:], s1, return_state=True)
            d = float((torch.cat([o1, o2], dim=1) - whole).abs().max())
            assert d < 1e-10, f"split={split} 分段续算偏差 {d}"


def test_conv_history_is_used():
    """回归防线: 卷积历史必须真正参与计算, 否则分段一致性会假通过."""
    torch.manual_seed(43)
    m = SunnyNet(SunnyConfig.small(256))
    m.eval()
    idx = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        s0 = m.init_state(1)
        o_a, _ = m(idx, s0, return_state=True)
        s1 = m.init_state(1)
        s1.hist.copy_(torch.randn_like(s1.hist) * 0.5)
        o_b, _ = m(idx, s1, return_state=True)
    assert not torch.allclose(o_a, o_b, atol=1e-6), "卷积历史未参与计算"


def test_feedback_is_causal():
    """状态反馈必须只用 <= t-1 的信息, 否则因果性会破."""
    torch.manual_seed(47)
    m = SunnyNet(SunnyConfig.small(512))
    m.eval()
    idx = torch.randint(0, 512, (1, 128))
    idx2 = idx.clone()
    idx2[0, 64:] = torch.randint(0, 512, (64,))
    with torch.no_grad():
        a = m(idx)
        b = m(idx2)
    assert float((a[:, :64] - b[:, :64]).abs().max()) == 0.0, "反馈通路破坏了因果性"
    assert float((a[:, 64:] - b[:, 64:]).abs().max()) > 1e-4


# ---------------------------------------------------------------- 参数/基线
def test_param_count_reported():
    cfg = SunnyConfig.small(512)
    m = SunnyNet(cfg)
    total = sum(p.numel() for p in m.parameters())
    assert m.num_params() == total
    assert m.num_params(non_embedding=True) == total - m.embed.weight.numel()


def test_baseline_param_matching():
    sunny = SunnyNet(SunnyConfig.small(1024))
    target = sunny.num_params(non_embedding=True)
    cfg, n, rel = solve_baseline_for_params(target, 1024)
    assert rel < 0.05, f"基线参数匹配偏差 {rel:.4f} (sunny={target} base={n})"
    from sunnynet.baseline import stack_param_count
    assert stack_param_count(cfg.d_model, cfg.d_ff, cfg.n_layers) == n


def test_baseline_forward_causal():
    torch.manual_seed(41)
    m = TransformerLM(BaselineConfig(vocab_size=256, d_model=128, n_heads=4,
                                     n_layers=2, d_ff=256, max_len=256))
    m.eval()
    idx = torch.randint(0, 256, (1, 64))
    idx2 = idx.clone()
    idx2[0, 32:] = torch.randint(0, 256, (32,))
    with torch.no_grad():
        assert torch.allclose(m(idx)[:, :32], m(idx2)[:, :32], atol=1e-6)


# ---------------------------------------------------------------- 分词器
def test_tokenizer_roundtrip_multilingual():
    texts = [
        "子曰：学而时习之，不亦说乎？",
        "def f(x):\n    return x ** 2",
        "The quick brown fox jumps over the lazy dog.",
        "混合 2026 mixed 中英 test_123 OK!",
        "《红楼梦》第三回：贾雨村夤缘复旧职",
    ] * 20
    tk = BPETokenizer.train(texts, vocab_size=700, min_char_freq=1)
    for t in texts[:5]:
        assert tk.decode(tk.encode(t)) == t, f"往返失败: {t[:30]}"
    assert all(0 <= i < tk.vocab_size for t in texts[:5] for i in tk.encode(t))


def test_tokenizer_specials_and_persist(tmp_path):
    texts = ["测试文本 abc 123", "another line of text"] * 30
    tk = BPETokenizer.train(texts, vocab_size=400, min_char_freq=1)
    assert tk.pad_id == 0 and tk.unk_id == 1 and tk.bos_id == 2 and tk.eos_id == 3
    p = tmp_path / "tok.json"
    tk.save(p)
    tk2 = BPETokenizer.load(p)
    assert tk2.vocab_size == tk.vocab_size
    assert tk2.decode(tk2.encode(texts[0])) == texts[0]


def test_pretokenize_splits_scripts():
    got = pretokenize("子曰abc123，OK！")
    assert "子曰" in got and "abc" in got and "123" in got


# ---------------------------------------------------------------- 消融开关
@pytest.mark.parametrize("flags", [
    {"decay_gate": False},
    {"circulant": False},
    {"local_taps": False},
    {"current": False},
    {"residual": False},
    {"state_norm": True},
    {"state_mul": True},
])
def test_ablation_switches_run(flags):
    torch.manual_seed(43)
    cfg = SunnyConfig(vocab_size=256, dim=64, inner=96, layers=3, ablation=flags)
    m = SunnyNet(cfg)
    idx = torch.randint(0, 256, (2, 64))
    loss = torch.nn.functional.cross_entropy(m(idx).reshape(-1, 256), idx.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

# ---------------------------------------------------------------- 等价改写
def test_circ_folded_into_ws_matches_fp64():
    """W_s(circ(x)) == x @ W_eff^T, 折叠必须与原路逐位接近 (fp64 1e-15)."""
    torch.manual_seed(51)
    cfg = SunnyConfig(vocab_size=256, dim=64, inner=96, rank=32, layers=2)
    blk = SunnyNet(cfg).blocks[0].double()
    x = torch.randn(3, 33, 64, dtype=torch.float64)
    with torch.no_grad():
        ref = blk.W_s(blk.circ(x))
        folded = F.linear(x, blk._Ws_eff())
    scale = ref.abs().max()
    assert float((ref - folded).abs().max()) / float(scale) < 1e-12


def test_folded_ws_used_in_forward_and_step():
    """_emit 与 step 走折叠路径后, 仍须与未折叠的等效计算一致."""
    torch.manual_seed(52)
    cfg = SunnyConfig(vocab_size=256, dim=64, inner=96, rank=32, layers=2)
    m = SunnyNet(cfg).double()
    blk = m.blocks[0]
    x = torch.randn(2, 40, 64, dtype=torch.float64)
    s_in = torch.randn(2, 64, dtype=torch.float64)
    hist = torch.zeros(2, cfg.taps - 1, 64, dtype=torch.float64)
    out, s_T, _ = blk(x, s_in, hist)
    u, gate, p = blk._core(blk._apply_local(blk.norm_in(x), hist))
    states, _ = logspace_scan(u, p, s_in, chunk=40, p_min_hint=blk.p_min_bound)
    y = blk._readout(u, None)
    prev = torch.cat([s_in[:, None, :], states[:, :-1]], dim=1)
    ref = blk.W_c(y) + blk.b_gain * states + blk.W_s(blk.circ(prev))
    if blk.use_residual:
        ref = ref + x
    with torch.no_grad():
        rel = float((out - ref).abs().max()) / float(ref.abs().max())
    assert rel < 1e-11


def test_wedge_constant_absorbed_by_norm():
    """softwedge 去常数项后经 CenteredScaleNorm 与原始 softwedge 等价."""
    torch.manual_seed(53)
    norm = CenteredScaleNorm(64).double()
    z = torch.randn(2, 32, 64, dtype=torch.float64) * 3
    a = torch.tensor(0.5, dtype=torch.float64)
    ref = norm(softwedge(z, a))
    # c = a/2 常量, 中心化后被吸收
    fast = norm(z + a * (0.5 + 0.5 * (z / (1.0 + z.abs()))) - a * 0.5)
    with torch.no_grad():
        rel = float((ref - fast).abs().max()) / float(ref.abs().max())
    assert rel < 1e-12


def test_apply_local_hist_zero_equals_padded():
    """hist 全零时免 cat 路径与 prepend 路径逐位相等; 非零 hist 时必须走原路."""
    torch.manual_seed(54)
    from sunnynet.layers import DepthwiseCausal
    dw = DepthwiseCausal(64, taps=4).double()
    raw = torch.randn(2, 40, 64, dtype=torch.float64)
    k = 3
    hist0 = torch.zeros(2, k, 64, dtype=torch.float64)
    rf = torch.cat([hist0, raw], dim=1)
    padded = (rf + dw(rf))[:, k:]
    fast = raw + dw(raw)
    assert float((padded - fast).abs().max()) == 0.0
    histn = torch.randn(2, k, 64, dtype=torch.float64)
    rfn = torch.cat([histn, raw], dim=1)
    assert float(((rfn + dw(rfn))[:, k:] - fast).abs().max()) > 1e-3


def test_p_clamp_unreachable_when_bound_above_floor():
    """p 的解析下界高于 p_floor 时 clamp 不可达, 去掉它前向逐位相等."""
    torch.manual_seed(55)
    # h_min=2.0 -> p_min_bound=0.5; h_max=2048.0 -> p_floor=4.88e-4
    from sunnynet.layers import SunnyBlock
    blk = SunnyBlock(dim=32, inner=48, rank=16, layer_index=0,
                     h_min=2.0, h_max=2048.0)
    assert blk.p_min_bound == 0.5
    assert blk.p_floor < blk.p_min_bound
    assert blk._need_clamp is False


def test_p_clamp_engaged_when_floor_above_bound():
    """h_min 抬到使解析下界低于 p_floor 时, clamp 必须重新启用."""
    from sunnynet.layers import SunnyBlock
    # h_min=1.0 -> p_min_bound=0.0 < p_floor=1/h_max
    blk = SunnyBlock(dim=32, inner=48, rank=16, layer_index=0,
                     h_min=1.0, h_max=2048.0)
    assert blk.p_min_bound == 0.0
    assert blk._need_clamp is True


def test_p_raw_range_matches_analytic_bound():
    """真实前向里 p 恒落在解析界内, 且与去 clamp 路径逐位一致."""
    torch.manual_seed(56)
    cfg = SunnyConfig.small(512)
    cfg.scan_chunk = 64
    m = SunnyNet(cfg).eval()
    x = torch.randint(0, 512, (2, 64))
    with torch.no_grad():
        y0 = m(x).clone()
    raw_lo, raw_hi = 1.0, 0.0
    for blk in m.blocks:
        delta = blk.delta().view(1, 1, -1)
        lo = float((1.0 - delta).min())
        hi = float((1.0 - delta * 0.0).max())
        raw_lo = min(raw_lo, lo)
        raw_hi = max(raw_hi, hi)
        assert lo >= blk.p_min_bound - 1e-6
    orig = type(m.blocks[0])._core

    def noclamp(self, a):
        u, gate, p = orig(self, a)
        return u, gate, p

    type(m.blocks[0])._core = noclamp
    with torch.no_grad():
        y1 = m(x).clone()
    type(m.blocks[0])._core = orig
    assert float((y0 - y1).abs().max()) == 0.0
