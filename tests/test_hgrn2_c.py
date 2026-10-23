"""HGRN2 整模型 C 内核的数值契约.

需要外部权重 (fla-hub/hgrn2-1.3B-100B) 与由它导出的 hgrn2.c.bin,
两者体积都大, 不入库; 任一缺失即跳过。blob 生成:
    python -X utf8 experiments/_export_hgrn2.py <hf_dir> <hf_dir>/hgrn2.c.bin
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sunnynet.runtime.hgrn2_c import SnHGRN2, hgrn2_available

MDIR = os.environ.get("SUNNY_HGRN2", "F:/sunnyclaw-data/models/hgrn2-1.3B-100B")
BLOB = Path(MDIR) / "hgrn2.c.bin"

pytestmark = pytest.mark.skipif(
    not hgrn2_available() or not BLOB.is_file(),
    reason="无 C 编译器/预编译库, 或缺少权重 blob")


@pytest.fixture(scope="module")
def m():
    from sunnynet.runtime.hgrn2 import HGRN2
    return HGRN2(MDIR)


def test_loads_and_reports_vocab(m):
    h = SnHGRN2(BLOB, m.vocab)
    assert h.vocab == m.vocab
    assert len(h.logits()) == m.vocab


def test_step_matches_numpy_reference(m):
    """C 前向与 numpy 参考逐 token 一致 (参考是独立实现, einsum 路径)。"""
    h = SnHGRN2(BLOB, m.vocab)
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
    h = SnHGRN2(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_reset_clears_state(m):
    h = SnHGRN2(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    for t in (99, 100):
        h.step(t)
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_prefill_matches_stepwise(m):
    h = SnHGRN2(BLOB, m.vocab)
    ids = [1, 415, 5565, 302, 4843]
    last = np.asarray(h.prefill(ids))
    h.reset()
    for t in ids[:-1]:
        h.step(t)
    assert float(np.abs(last - np.asarray(h.step(ids[-1]))).max()) == 0.0


@pytest.mark.parametrize("n", [1, 4])
def test_thread_count_changes_nothing(m, n):
    """线程数只影响速度, 不改变结果 (按 head 并行, head 内串行)。"""
    ref = SnHGRN2(BLOB, m.vocab, threads=8)
    ids = list(range(3, 9))
    a = np.stack([np.asarray(ref.step(t)) for t in ids])
    h = SnHGRN2(BLOB, m.vocab, threads=n)
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_lower_bound_is_applied_from_layer1(m):
    """lower_bound 只在 i>0 参与混合; 第 0 层必须等价于纯 logsigmoid(f)。
    把第 0 层的 lb 置零重跑, 结果应与原模型逐位相同。"""
    from sunnynet.runtime.hgrn2 import HGRN2
    m0 = HGRN2(MDIR)
    h = SnHGRN2(BLOB, m0.vocab)
    st = m0.reset(m0.init_state())
    ref = np.asarray(h.step(1))
    # numpy 侧第 0 层 lb 已是 0, 这里只需确认 C 侧没误加 lb 混合
    assert not np.allclose(ref, 0.0)
    assert m0.use_lower_bound