"""AVX-512 / NEON 内核的可编译性与正确性测试.

本机 (i7-8750H) 不支持 AVX-512 也无 AArch64, 因此:
  - AVX-512: 验证「能编译出 zmm 指令」+「不支持时拒绝加载」; 数值验证仅在
    支持 AVX512F 的机器上运行。
  - NEON: 验证 zig 交叉编译产出 AArch64 ELF 且含 4-lane NEON 指令;
    数值验证仅在 AArch64 上运行。
这与本仓"不伪造结论"的纪律一致: 不能执行的路径只报告编译事实。
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sunnynet.kernels import simd

HERE = Path(__file__).resolve().parents[1]
KSRC = HERE / "sunnynet" / "kernels"


def _gcc():
    from shutil import which
    env = os.environ.get("SUNNY_W64")
    if env:
        p = Path(env) / "bin" / "gcc.exe"
        if p.exists():
            return str(p)
    return which("gcc")


NEED_GCC = pytest.mark.skipif(_gcc() is None, reason="无 C 编译器")
HAS_AVX512 = simd.avx512_supported()
IS_AARCH64 = simd.arch_name() == "aarch64"
# AVX-512 源码只在 x86-64 上能编译 (AArch64 无 mavx512* 特性)
IS_X86 = simd.arch_name() == "x86_64"


# ------------------------------------------------------------- 架构探测
def test_arch_name_known():
    assert simd.arch_name() in ("x86_64", "aarch64", "arm")


def test_avx512_probe_matches_known_host():
    """本机 CPU 是 Coffee Lake, 必然不支持 AVX512F."""
    got = simd.avx512_supported()
    assert isinstance(got, bool)
    if simd.arch_name() != "x86_64":
        assert got is False


# ------------------------------------------------------- AVX-512 编译
@pytest.mark.skipif(not IS_X86, reason="AVX-512 源码只能在 x86-64 上编译")
@NEED_GCC
def test_avx512_compiles_with_zmm(tmp_path):
    """AVX-512 源码必须能编译, 且产出含 zmm 寄存器的目标文件."""
    gcc = _gcc()
    obj = tmp_path / "scan512.o"
    cmd = [gcc, "-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma",
           "-funroll-loops", "-std=c11", "-c",
           str(KSRC / "scan_avx512.c"), "-o", str(obj)]
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, (r.stderr or b"").decode("utf-8", "replace")[-600:]
    assert obj.exists() and obj.stat().st_size > 0

    # 反汇编找 zmm 与 vfmadd
    od = Path(gcc).parent / ("objdump.exe" if os.name == "nt" else "objdump")
    assert od.exists(), "缺 objdump"
    r2 = subprocess.run([str(od), "-d", str(obj)], capture_output=True)
    dis = (r2.stdout or b"").decode("utf-8", errors="replace")
    assert "zmm" in dis, "未生成 zmm 寄存器指令"
    assert "vfmadd" in dis.lower(), "未生成融合乘加"


@pytest.mark.skipif(HAS_AVX512, reason="仅在无 AVX-512 的机器上验证拒绝加载")
@NEED_GCC
def test_avx512_refuses_load_on_unsupported_cpu():
    with pytest.raises(RuntimeError, match="不支持 AVX512F"):
        simd.load_avx512(force_rebuild=True)


@pytest.mark.skipif(not HAS_AVX512, reason="本机不支持 AVX512F, 跳过数值验证")
@NEED_GCC
def test_avx512_scan_matches_reference():
    import ctypes
    lib = simd.load_avx512(force_rebuild=True)
    assert lib.sn512_cpu_supported() == 1
    rng = np.random.default_rng(41)
    B, T, N = 2, 5, 20          # N=20 非 16 倍数, 触发掩码尾处理
    p = rng.random((B, T, N), dtype=np.float32)
    nu = rng.standard_normal((B, T, N)).astype(np.float32)
    s0 = rng.standard_normal((B, N)).astype(np.float32)
    out = np.zeros_like(p)
    send = np.zeros_like(s0)
    PF = ctypes.POINTER(ctypes.c_float)
    lib.scan512_f32(p.ctypes.data_as(PF), nu.ctypes.data_as(PF),
                    s0.ctypes.data_as(PF), out.ctypes.data_as(PF),
                    send.ctypes.data_as(PF), B, T, N)
    # 参考: 逐通道递推
    ref = np.zeros_like(p)
    for b in range(B):
        s = s0[b].copy()
        for t in range(T):
            s = p[b, t] * s + (1.0 - p[b, t]) * nu[b, t]
            ref[b, t] = s
    assert np.abs(out - ref).max() < 1e-5
    assert np.abs(send - ref[:, -1, :]).max() < 1e-5


# ---------------------------------------------------------- NEON 交叉编译
def _find_zig():
    return simd._find_zig()


@NEED_GCC
@pytest.mark.skipif(_find_zig() is None, reason="未安装 zig, 无法交叉编译")
def test_neon_cross_compiles_aarch64(tmp_path):
    """zig 交叉编译必须产出 AArch64 ELF64 共享库."""
    out = simd.cross_compile_neon(tmp_path / "neon.so")
    assert out.exists() and out.stat().st_size > 0
    data = out.read_bytes()
    assert data[:4] == b"\x7fELF"
    assert data[4] == 2, "应为 ELF64"
    e_machine = struct.unpack_from("<H", data, 18)[0]
    assert e_machine == 183, f"应指向 AArch64, 实际 {e_machine}"


@NEED_GCC
@pytest.mark.skipif(_find_zig() is None, reason="未安装 zig")
def test_neon_binary_has_vector_instructions(tmp_path):
    """交叉编译产物中的 .text 必须含 4-lane 单精度 NEON 指令."""
    capstone = pytest.importorskip("capstone")
    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN

    out = simd.cross_compile_neon(tmp_path / "neon.so")
    data = out.read_bytes()
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shent = struct.unpack_from("<H", data, 0x3a)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3c)[0]
    e_shstrndx = struct.unpack_from("<H", data, 0x3e)[0]

    def sh(i):
        o = e_shoff + i * e_shent
        name, typ, fl, addr, off, size = struct.unpack_from("<IIQQQQ", data, o)
        return name, addr, off, size

    _, _, st_off, st_size = sh(e_shstrndx)

    def sname(n):
        e = data.index(b"\x00", st_off + n)
        return data[st_off + n:e].decode()

    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    fmla_4s = fmul_4s = 0
    for i in range(e_shnum):
        nm_idx, addr, off, size = sh(i)
        if sname(nm_idx) != ".text":
            continue
        for ins in md.disasm(data[off:off + size], addr):
            ops = ins.op_str
            if ins.mnemonic == "fmla" and ".4s" in ops:
                fmla_4s += 1
            if ins.mnemonic == "fmul" and ".4s" in ops:
                fmul_4s += 1
    assert fmla_4s + fmul_4s > 0, f"未检出 4-lane NEON 向量指令 (fmla={fmla_4s}, fmul={fmul_4s})"


# ----------------------------------------------------------- NEON 执行
@pytest.mark.skipif(not IS_AARCH64, reason="NEON 执行验证需 AArch64 主机")
@NEED_GCC
def test_neon_scan_matches_reference():
    import ctypes
    lib = simd.load_neon(force_rebuild=True)
    rng = np.random.default_rng(42)
    B, T, N = 2, 5, 20
    p = rng.random((B, T, N), dtype=np.float32)
    nu = rng.standard_normal((B, T, N)).astype(np.float32)
    s0 = rng.standard_normal((B, N)).astype(np.float32)
    out = np.zeros_like(p)
    send = np.zeros_like(s0)
    PF = ctypes.POINTER(ctypes.c_float)
    lib.scan_neon_f32(p.ctypes.data_as(PF), nu.ctypes.data_as(PF),
                      s0.ctypes.data_as(PF), out.ctypes.data_as(PF),
                      send.ctypes.data_as(PF), B, T, N)
    ref = np.zeros_like(p)
    for b in range(B):
        s = s0[b].copy()
        for t in range(T):
            s = p[b, t] * s + (1.0 - p[b, t]) * nu[b, t]
            ref[b, t] = s
    assert np.abs(out - ref).max() < 1e-5


@pytest.mark.skipif(IS_AARCH64, reason="仅在非 AArch64 主机上验证明确报错")
def test_neon_load_refuses_on_x86():
    with pytest.raises(RuntimeError, match="只能在 AArch64 上执行"):
        simd.load_neon(force_rebuild=True)