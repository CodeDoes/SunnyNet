"""Mamba-2 整模型 C 内核的数值契约.

需要外部权重 (state-spaces/mamba2-130m) 与由它导出的 mamba2.c.bin,
两者体积都大, 不入库; 任一缺失即跳过。blob 生成:
    python -X utf8 experiments/_export_mamba2.py <hf_dir> <hf_dir>/mamba2.c.bin
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sunnynet.runtime.mamba2_c import SnMamba2, mamba2_available

MDIR = os.environ.get("SUNNY_MAMBA2",
                      str(Path(__file__).resolve().parents[1]
                          / "models" / "mamba2-130m"))
BLOB = Path(MDIR) / "mamba2.c.bin"

pytestmark = pytest.mark.skipif(
    not mamba2_available() or not BLOB.is_file(),
    reason="无 C 编译器/预编译库, 或缺少权重 blob")


@pytest.fixture(scope="module")
def m():
    from sunnynet.runtime.mamba2 import Mamba2
    return Mamba2(MDIR)


def test_loads_and_reports_vocab(m):
    h = SnMamba2(BLOB, m.vocab)
    assert h.vocab == m.vocab
    assert len(h.logits()) == m.vocab


def test_step_matches_numpy_reference(m):
    """C 前向与 numpy 参考逐 token 一致 (参考是独立实现, einsum 路径)。"""
    h = SnMamba2(BLOB, m.vocab)
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
    h = SnMamba2(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_reset_clears_state(m):
    """reset 后重跑必须与首次逐位相同 (状态确实清零)。"""
    h = SnMamba2(BLOB, m.vocab)
    ids = list(range(5, 15))
    a = np.stack([np.asarray(h.step(t)) for t in ids])
    for t in (99, 100):
        h.step(t)
    h.reset()
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0


def test_prefill_matches_stepwise(m):
    h = SnMamba2(BLOB, m.vocab)
    ids = [1, 415, 5565, 302, 4843]
    last = np.asarray(h.prefill(ids))
    h.reset()
    for t in ids[:-1]:
        h.step(t)
    assert float(np.abs(last - np.asarray(h.step(ids[-1]))).max()) == 0.0


@pytest.mark.parametrize("n", [1, 4])
def test_thread_count_changes_nothing(m, n):
    """线程数只影响速度, 不改变结果 (逐输出并行, 无跨线程归约)。"""
    ref = SnMamba2(BLOB, m.vocab, threads=8)
    ids = list(range(3, 9))
    a = np.stack([np.asarray(ref.step(t)) for t in ids])
    h = SnMamba2(BLOB, m.vocab, threads=n)
    b = np.stack([np.asarray(h.step(t)) for t in ids])
    assert float(np.abs(a - b).max()) == 0.0