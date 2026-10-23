"""GLA 整模型 C 内核的数值契约.

需要外部权重 (fla-org 的 GLA 340M) 与由它导出的 gla.c.bin,
两者体积都大, 不入库; 任一缺失即跳过。blob 生成:
    python -X utf8 experiments/_export_gla.py <hf_dir> <hf_dir>/gla.c.bin
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sunnynet.runtime.gla_c import SnGLA, gla_available

MDIR = os.environ.get("SUNNY_GLA",
                      str(Path(__file__).resolve().parents[1]
                          / "models" / "gla-340M"))
BLOB = Path(MDIR) / "gla.c.bin"

pytestmark = pytest.mark.skipif(
    not gla_available() or not BLOB.is_file(),
    reason="无 C 编译器/预编译库, 或缺少权重 blob")


@pytest.fixture(scope="module")
def m():
    from sunnynet.runtime.gla import GLA
    return GLA(MDIR)


def test_loads_and_reports_vocab(m):
    h = SnGLA(BLOB, m.vocab)
    assert h.vocab == m.vocab
    assert len(h.logits()) == m.vocab


def test_step_matches_numpy_reference(m):
    """C 前向与 numpy 参考逐 token 一致 (参考是独立实现, einsum 路径)。"""
    h = SnGLA(BLOB, m.vocab)
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
    h = SnGLA(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_reset_clears_state(m):
    """reset 后重跑必须与首次逐位相同 (状态确实清零)。"""
    h = SnGLA(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    for t in (99, 100):
        h.step(t)
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_prefill_matches_stepwise(m):
    h = SnGLA(BLOB, m.vocab)
    ids = [1, 415, 5565, 302, 4843]
    last = np.asarray(h.prefill(ids))
    h.reset()
    for t in ids[:-1]:
        h.step(t)
    assert float(np.abs(last - np.asarray(h.step(ids[-1]))).max()) == 0.0


@pytest.mark.parametrize("n", [1, 4])
def test_thread_count_changes_nothing(m, n):
    """线程数只影响速度, 不改变结果 (按 head 并行, head 内串行)。"""
    ref = SnGLA(BLOB, m.vocab, threads=8)
    ids = list(range(3, 9))
    a = np.stack([np.asarray(ref.step(t)) for t in ids])
    h = SnGLA(BLOB, m.vocab, threads=n)
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_head_wise_value_is_not_shared(m):
    """回归: gla_scan 曾把每个 head 都读 v 的第 0 段, 于是所有 head 共用同一
    value 向量。症状是单步 readout 差 96% 而各投影完全正确。
    判据: 不同 head 的 S 行在收到不同 v 时应产生不同的读出。"""
    h = SnGLA(BLOB, m.vocab)
    st = m.reset(m.init_state())
    ref = m.step(1, st)
    c = np.asarray(h.step(1))
    assert float(np.abs(c - ref).max()) < 1e-4, float(np.abs(c - ref).max())