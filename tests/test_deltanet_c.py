"""DeltaNet 整模型 C 内核的数值契约.

需要外部权重 (fla-org 的 DeltaNet 340M) 与由它导出的 deltanet.c.bin,
两者体积都大, 不入库; 任一缺失即跳过。blob 生成:
    python -X utf8 experiments/_export_deltanet.py <hf_dir> <hf_dir>/deltanet.c.bin
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sunnynet.runtime.deltanet_c import SnDeltaNet, deltanet_available

MDIR = os.environ.get("SUNNY_DELTANET",
                      str(Path(__file__).resolve().parents[1]
                          / "models" / "deltanet-pure"))
BLOB = Path(MDIR) / "deltanet.c.bin"
BLOB4 = Path(MDIR) / "deltanet.q4.bin"

pytestmark = pytest.mark.skipif(
    not deltanet_available() or not BLOB.is_file(),
    reason="无 C 编译器/预编译库, 或缺少权重 blob")

q4mark = pytest.mark.skipif(
    not deltanet_available() or not BLOB4.is_file(),
    reason="缺少 INT4 blob (deltanet.q4.bin)")


@pytest.fixture(scope="module")
def m():
    from sunnynet.runtime.deltanet import DeltaNet
    return DeltaNet(MDIR)


def test_loads_and_reports_vocab(m):
    h = SnDeltaNet(BLOB, m.vocab)
    assert h.vocab == m.vocab
    assert len(h.logits()) == m.vocab


def test_step_matches_numpy_reference(m):
    """C 前向与 numpy 参考逐 token 一致 (参考是独立实现, einsum 路径)。"""
    h = SnDeltaNet(BLOB, m.vocab)
    st = m.reset(m.init_state())
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 30))
    dmax, am = 0.0, 0
    for t in ids:
        ref = m.step(int(t), st)
        c = np.asarray(h.step(int(t)))
        dmax = max(dmax, float(np.abs(c - ref).max()))
        am += int(c.argmax() == ref.argmax())
    assert dmax < 1e-3, f"max|diff|={dmax:.3e}"
    assert am == len(ids), f"argmax {am}/{len(ids)}"


def test_repeated_run_is_bit_stable(m):
    h = SnDeltaNet(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_reset_clears_state(m):
    """reset 后重跑必须与首次逐位相同 (状态确实清零)。"""
    h = SnDeltaNet(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    for t in (99, 100):
        h.step(t)
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_prefill_matches_stepwise(m):
    h = SnDeltaNet(BLOB, m.vocab)
    ids = [1, 415, 5565, 302, 4843]
    last = np.asarray(h.prefill(ids))
    h.reset()
    for t in ids[:-1]:
        h.step(t)
    assert float(np.abs(last - np.asarray(h.step(ids[-1]))).max()) == 0.0


@pytest.mark.parametrize("n", [1, 4])
def test_thread_count_changes_nothing(m, n):
    """线程数只影响速度, 不改变结果。"""
    ref = SnDeltaNet(BLOB, m.vocab, threads=8)
    ids = list(range(3, 9))
    a = np.stack([np.asarray(ref.step(t)) for t in ids])
    h = SnDeltaNet(BLOB, m.vocab, threads=n)
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


@q4mark
def test_int4_loads_and_stays_finite(m):
    """Q4 blob 走带标签的解析路径; 短卷积核与 b_proj 仍为 fp32。
    相对误差上限 0.6 (4-bit 理论 ~0.10, 实测中位 0.24), 只挡解析错位。"""
    h32 = SnDeltaNet(BLOB, m.vocab)
    h4 = SnDeltaNet(BLOB4, m.vocab)
    assert h4.vocab == m.vocab
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h32.step(t)) for t in ids])
    b = np.stack([np.asarray(h4.step(t)) for t in ids])
    assert np.isfinite(b).all()
    d = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-9)
    assert 0.0 < d < 0.6, f"相对 max|diff|={d:.3e}"
    agree = sum(int(a[i].argmax() == b[i].argmax()) for i in range(len(ids)))
    assert agree >= len(ids) // 2, f"argmax 一致性 {agree}/{len(ids)}"


@q4mark
def test_int4_repeated_run_is_bit_stable():
    h = SnDeltaNet(BLOB4, 32000)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0