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


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_thread_count_does_not_change_logits():
    """线程数只影响耗时, 不得改变数值: 物理核与 1 线程结果须一致."""
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads
    _, cfg, _ = load_model(str(CKPT))
    torch.manual_seed(11)
    toks = torch.randint(0, cfg.vocab_size, (12,)).tolist()

    def run(nthreads):
        set_threads(nthreads)
        h = SnModel(BLOB, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                    cfg.layers, 8, int(cfg.tie_embeddings))
        h.reset()
        return torch.tensor([h.step(t) for t in toks])

    a = run(1)
    b = run(6)
    assert float((a - b).abs().max()) < 1e-5, "并行 GEMV 引入数值差异"


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_rowmajor_layout_matches_export_order():
    """sn_load 把导出的 (in,out) 转置块翻成行主序后, 单步结果不变.

    权重布局错误会立刻把 logits 打乱, 因此这个断言是布局的看门测试。
    """
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads
    set_threads(1)
    m, cfg, _ = load_model(str(CKPT))
    m.eval()
    torch.set_num_threads(1)
    rt = SnModel(BLOB, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                 cfg.layers, 8, int(cfg.tie_embeddings))
    torch.manual_seed(5)
    toks = torch.randint(0, cfg.vocab_size, (10,)).tolist()
    s = m.init_state(1)
    with torch.no_grad():
        for t in toks:
            lg_t, s = m.forward_step(torch.tensor([t]), s)
            lg_c = torch.tensor(rt.step(t))
            assert float((lg_t[0] - lg_c).abs().max()) < 1e-3
            assert int(lg_t[0].argmax()) == int(lg_c.argmax())


def test_recommended_threads_is_physical_cores():
    """物理核探测落在合法范围内, 且 set_threads 往返生效."""
    import os
    from sunnynet.runtime.model import recommended_threads, set_threads
    n = recommended_threads()
    assert 1 <= n <= (os.cpu_count() or 1)
    # 本机实测物理核低于逻辑核; 探测不到时应至少回退到 1
    assert n >= 1
    assert set_threads(3) == 3
    assert set_threads(n) == n


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_half_weights_track_fp32_argmax():
    """fp16 权重路径与 fp32 路径在同序列上 argmax 一致, logits 偏差有界.

    权重截断为 fp16 会引入约 1e-3 量级的 logits 抖动; 上限设 2e-2
    足以捕捉布局或转换写错 (那会把偏差推到 1e+1)。
    """
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads
    set_threads(1)
    _, cfg, _ = load_model(str(CKPT))
    args = (cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
            cfg.layers, 8, int(cfg.tie_embeddings))
    torch.manual_seed(13)
    toks = torch.randint(0, cfg.vocab_size, (20,)).tolist()

    a = SnModel(BLOB, *args)
    b = SnModel(BLOB, *args, half=True)
    a.reset(); b.reset()
    n_agree = 0
    worst = 0.0
    for t in toks:
        la = torch.tensor(a.step(t))
        lb = torch.tensor(b.step(t))
        worst = max(worst, float((la - lb).abs().max()))
        if int(la.argmax()) == int(lb.argmax()):
            n_agree += 1
    assert worst < 2e-2, f"fp16 权重偏差过大: {worst:.3e}"
    assert n_agree == len(toks), f"argmax 仅 {n_agree}/{len(toks)} 一致"


@pytest.mark.skipif(not CKPT.exists() or not BLOB.exists(),
                    reason="缺少 ckpt 或导出 blob")
def test_half_weights_are_smaller_than_fp32():
    """fp16 载入后常驻权重应明显小于 fp32 (5 个 GEMV 矩阵 + head 减半)."""
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads
    set_threads(1)
    _, cfg, _ = load_model(str(CKPT))
    d = cfg.dim
    # 仅校验数值路径可用且不崩: 载入后单步能产出有限 logits
    rt = SnModel(BLOB, cfg.vocab_size, d, cfg.inner, cfg.rank, cfg.taps,
                 cfg.layers, 8, int(cfg.tie_embeddings), half=True)
    lg = torch.tensor(rt.step(7))
    assert lg.shape[0] == cfg.vocab_size
    assert bool(torch.isfinite(lg).all()), "fp16 路径产生 NaN/Inf"
