"""INT4 分组量化与量化 GEMV 的自检."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from sunnynet.runtime import quant as Q  # noqa: E402


def _have():
    try:
        return Q.quant_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have(), reason="无 C 编译器 (设 SUNNY_W64)")


def test_quant_roundtrip_within_step():
    """反量化结果与 x 的逐元素误差不超过该 block 的 scale 一半."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 256)) * 2.0).astype(np.float32)
    q, s = Q.quant_i4(x, 64)
    xd = Q.dequant_i4(q, s, 256, 64)
    blk = s.reshape(8, 4)[:, :, None]
    assert np.all(np.abs(x - xd).reshape(8, 4, 64) <= blk / 2 + 1e-6)


def test_gemv_i4_matches_explicit_integer_formula():
    """量化 GEMV 的整数路径与显式整数求和逐位一致."""
    rng = np.random.default_rng(1)
    out, in_dim, block = 64, 256, 64
    W = rng.standard_normal((out, in_dim)).astype(np.float32)
    x = rng.standard_normal(in_dim).astype(np.float32)
    q, s = Q.quant_i4(W, block)

    # 显式还原: 打包布局为 字节 k 低半字节 = 组内第 k 个, 高半字节 = 第 (k + block/2) 个
    half = block // 2
    nb = in_dim // block
    Wint = np.empty((out, in_dim), np.int32)
    for o in range(out):
        for b in range(nb):
            for k in range(half):
                byte = int(q[o, b * half + k])
                Wint[o, b * block + k] = (byte & 0x0F) - 8
                Wint[o, b * block + half + k] = (byte >> 4) - 8

    # 激活按同 block 量化到 INT8
    xq = np.empty(in_dim, np.int8)
    xs = np.empty(nb, np.float32)
    for b in range(nb):
        seg = x[b * block:(b + 1) * block]
        amax = float(np.abs(seg).max())
        sc = amax / 127.0 if amax > 0 else 1.0
        xq[b * block:(b + 1) * block] = np.clip(np.rint(seg / sc), -127, 127).astype(np.int8)
        xs[b] = sc

    y_ref = np.empty(out, np.float32)
    for o in range(out):
        acc = 0.0
        for b in range(nb):
            seg = slice(b * block, (b + 1) * block)
            acc += s[o, b] * xs[b] * float((Wint[o, seg] * xq[seg].astype(np.int32)).sum())
        y_ref[o] = acc

    y = Q.gemv_i4(x, q, s, block)
    assert float(np.abs(y - y_ref).max()) < 1e-4


def test_gemv_i4_error_bound_vs_fp32():
    """对 fp32 GEMV 的相对误差在 4-bit 预期范围内 (<0.2)."""
    rng = np.random.default_rng(2)
    W = rng.standard_normal((256, 768)).astype(np.float32) * 0.05
    x = rng.standard_normal(768).astype(np.float32)
    q, s = Q.quant_i4(W, 64)
    yq = Q.gemv_i4(x, q, s, 64)
    yf = Q.gemv_f32(x, W)
    rel = float(np.linalg.norm(yq - yf) / np.linalg.norm(yf))
    assert rel < 0.2, rel


def test_quant_block_sizes():
    """block 32/64/128 都能跑通且误差量级相当."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((128, 1024)).astype(np.float32) * 0.05
    x = rng.standard_normal(1024).astype(np.float32)
    yf = Q.gemv_f32(x, W)
    rels = []
    for block in (32, 64, 128):
        q, s = Q.quant_i4(W, block)
        yq = Q.gemv_i4(x, q, s, block)
        rels.append(float(np.linalg.norm(yq - yf) / np.linalg.norm(yf)))
    assert max(rels) < 0.2
    assert max(rels) / min(rels) < 3.0