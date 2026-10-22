"""稳定 C ABI 的契约测试.

覆盖: 版本查询与兼容性判定、能力位、错误码路径、递推/量化数值正确性、
以及 RWKV-7 经 ABI 与经内部句柄的结果一致性。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from sunnynet.runtime import abi  # noqa: E402

pytestmark = pytest.mark.skipif(
    not abi.abi_available(), reason="无 C 编译器, 跳过 ABI 测试")

BLOB = os.environ.get(
    "SUNNY_RWKV7_BLOB",
    str(ROOT / "experiments" / "gguf" / "rwkv7-191M-world-bf16.runtime.bin"))


def test_version_matches_header():
    assert abi.version() == abi.ABI_VERSION == 0x10000
    assert abi.version_string() == "1.0.0"


def test_caps_include_core():
    names = abi.cap_names()
    for want in ("recur_scan", "recur_step", "quant_i8", "threads"):
        assert want in names


def test_check_compat_and_incompat():
    assert abi.check(abi.ABI_MAJOR) == abi.STATUS_OK
    rc = abi.check(abi.ABI_MAJOR + 1)
    assert rc == -5  # SN_E_UNSUPPORTED


def test_status_messages_are_defined():
    for code in (0, -1, -2, -3, -4, -5):
        assert abi.status_str(code)


def test_recur_step_matches_reference():
    lib = abi.load_abi()
    rng = np.random.default_rng(31)
    B, N = 3, 16
    a = rng.random((B, N), dtype=np.float32) * 0.9
    b = rng.standard_normal((B, N)).astype(np.float32)
    s0 = rng.standard_normal((B, N)).astype(np.float32)
    ref = a * s0 + b

    s = s0.copy()
    rc = lib.sn_abi_recur_step(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, N, 0)
    assert rc == abi.STATUS_OK
    assert np.abs(s - ref).max() < 1e-6


def test_recur_step_gated_semantics():
    lib = abi.load_abi()
    rng = np.random.default_rng(32)
    B, N = 2, 8
    a = rng.random((B, N), dtype=np.float32)
    u = rng.standard_normal((B, N)).astype(np.float32)
    s0 = rng.standard_normal((B, N)).astype(np.float32)
    ref = a * s0 + (1.0 - a) * u
    s = s0.copy()
    lib.sn_abi_recur_step(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, N, 1)
    assert np.abs(s - ref).max() < 1e-6


def test_invalid_args_return_einval():
    lib = abi.load_abi()
    a = np.zeros((2, 8), dtype=np.float32)
    pa = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 空指针
    assert lib.sn_abi_recur_step(None, pa, pa, 2, 8, 0) == -1
    # 负维度
    assert lib.sn_abi_recur_step(pa, pa, pa, 2, -8, 0) == -1
    # set_threads 非正
    assert lib.sn_abi_set_threads(0) == -1


def test_quant_roundtrip_via_abi():
    lib = abi.load_abi()
    rng = np.random.default_rng(33)
    B, N, block = 3, 64, 8
    x = rng.standard_normal((B, N)).astype(np.float32)
    q = np.empty((B, N), dtype=np.int8)
    sc = np.empty((B, N // block), dtype=np.float32)
    rc = lib.sn_abi_quant_i8(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q.ctypes.data_as(ctypes.c_void_p),
        sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, N, block)
    assert rc == abi.STATUS_OK
    y = np.empty((B, N), dtype=np.float32)
    lib.sn_abi_dequant_i8(
        q.ctypes.data_as(ctypes.c_void_p),
        sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, N, block)
    # int8 量化误差: 每 block 内步长约为 max|.|/127, 留足余量
    assert np.abs(x - y).max() < 0.1


@pytest.mark.skipif(not os.path.exists(BLOB), reason="RWKV-7 blob 不在本机")
def test_rwkv7_abi_matches_internal():
    """经稳定 ABI 与经内部句柄的 logits 必须逐位一致."""
    from sunnynet.runtime.rwkv7_c import SnRWKV7
    from sunnynet.runtime.rwkv7 import RWKV7

    internal = SnRWKV7(BLOB, 65536)
    external = abi.Rwkv7ABI(BLOB)
    assert external.vocab == 65536

    ids = [1, 42, 1000, 500, 77]
    internal.reset()
    a = np.array(internal.prefill(ids), dtype=np.float32)

    b = np.array(external.prefill(ids), dtype=np.float32)

    assert np.abs(a - b).max() == 0.0
    assert a.argmax() == b.argmax()

    # 再单步一次, 两侧必须仍然一致
    internal.reset()
    external.reset()
    for t in ids:
        internal.step_raw(t)
    external.prefill(ids)
    nxt = int(np.argmax(internal.logits()))
    la = np.array(internal.step(nxt), dtype=np.float32)
    lb = np.array(external.step(nxt), dtype=np.float32)
    assert np.abs(la - lb).max() == 0.0

    external.close()