"""SunnyNet 手写内核的编译与加载 (AVX2/FMA, w64devkit GCC).

编译器不在系统 PATH 时用 SUNNY_W64 环境变量指向 w64devkit 根目录。
首次调用自动编译, 之后复用 dll。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["load_scan_kernel", "load_kernel", "kernel_available",
           "set_kernel_threads", "TOOLCHAIN_HINT"]

HERE = Path(__file__).resolve().parent
SRC = HERE / "scan_avx2.c"
CORE_SRC = HERE / "core_avx2.c"
STEP_SRC = HERE / "step_avx2.c"
SOURCES = [SRC, CORE_SRC, STEP_SRC]
BUILD = HERE / "_build"

_COMMON_FLAGS = ["-O3", "-mavx2", "-mfma", "-march=x86-64-v3", "-funroll-loops"]


def _compile_flags() -> list[str]:
    """Windows (mingw) 要 `-static -shared`; Linux 的 GNU ld 不接受该组合,
    且必须 `-fPIC`, 否则报 "attempted static link of dynamic object" 与
    "recompile with -fPIC"。"""
    f = list(_COMMON_FLAGS)
    if sys.platform == "win32":
        f += ["-fopenmp", "-static", "-shared"]
    else:
        f += ["-fopenmp", "-fPIC", "-shared"]
    return f


DLL = BUILD / ("sunnynet_kernels.dll" if sys.platform == "win32"
               else "sunnynet_kernels.so")

TOOLCHAIN_HINT = (
    "未找到 C 编译器。设置环境变量 SUNNY_W64 指向 w64devkit 根目录, 例如:\n"
    "  set SUNNY_W64=<w64devkit 解压目录>"
)


def _find_gcc() -> str | None:
    from shutil import which
    env = os.environ.get("SUNNY_W64")
    if env:
        for name in ("gcc.exe", "gcc"):
            c = Path(env) / "bin" / name
            if c.exists():
                return str(c)
    return which("gcc")


def _cpu_has_avx2() -> bool:
    """内核用 AVX2/FMA, 无此指令集的机器上运行会非法指令崩溃。"""
    if os.name != "nt":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                return "avx2" in f.read()
        except OSError:
            return False
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _PF(ctypes.Structure):
            _fields_ = [("wProcessorArchitecture", wt.WORD),
                        ("wReserved", wt.WORD),
                        ("dwPageSize", wt.DWORD),
                        ("lpMinimumApplicationAddress", ctypes.c_void_p),
                        ("lpMaximumApplicationAddress", ctypes.c_void_p),
                        ("dwActiveProcessorMask", ctypes.c_void_p),
                        ("dwNumberOfProcessors", wt.DWORD),
                        ("dwProcessorType", wt.DWORD),
                        ("dwAllocationGranularity", wt.DWORD),
                        ("wProcessorLevel", wt.WORD),
                        ("wProcessorRevision", wt.WORD)]

        k = ctypes.windll.kernel32
        k.IsProcessorFeaturePresent.argtypes = [wt.DWORD]
        k.IsProcessorFeaturePresent.restype = wt.BOOL
        for feat in (40, 39):  # AVX2 = 40, AVX = 39 (PF_AVX2_INSTRUCTIONS_AVAILABLE)
            if k.IsProcessorFeaturePresent(feat):
                return True
        # 回退: 用 CPUID 查 AVX2 (leaf 7, EBX bit 5)
        return _cpuid_avx2()
    except Exception:
        return False


def _cpuid_avx2() -> bool:
    try:
        import numpy as np
        # numpy 若以 AVX2 构建则本机必然支持; 这只是保守回退
        return "avx2" in " ".join(np.__config__.CONFIG.get("SIMD Extensions", [])).lower()
    except Exception:
        return False


def kernel_available() -> bool:
    return _find_gcc() is not None and _cpu_has_avx2()


def _compile(gcc: str) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = [gcc] + _compile_flags()
    cmd += [str(s) for s in SOURCES]
    cmd += ["-o", str(DLL)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        raise RuntimeError(f"kernel 编译失败: {tail}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    dll_t = DLL.stat().st_mtime
    return any(s.stat().st_mtime > dll_t for s in SOURCES)


def load_kernel(force_rebuild: bool = False):
    """按 autograd.Function 里用到的签名批量设好 argtypes."""
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError(TOOLCHAIN_HINT)
    if force_rebuild or _needs_rebuild():
        _compile(gcc)
    lib = ctypes.CDLL(str(DLL))
    V = ctypes.c_void_p
    I = ctypes.c_int
    lib.scan_f32.argtypes = [V, V, V, V, V, I, I, I]
    lib.scan_bwd_f32.argtypes = [V, V, V, V, V, V, V, V, I, I, I]
    lib.core_fwd_f32.argtypes = [V, V, V, V, V, V, I, I, I]
    lib.core_bwd_f32.argtypes = [V, V, V, V, V, V, V, V, V, V, V, I, I, I]
    lib.dw_step_f32.argtypes = [V, V, V, V, I, I, I]
    lib.step_core_f32.argtypes = [V, V, V, V, V, I, I]
    lib.sn_set_threads.argtypes = [I]
    return lib


def load_scan_kernel(force_rebuild: bool = False):
    """返回 (lib, proto) —— proto 已设好 argtypes."""
    return load_kernel(force_rebuild=force_rebuild)


def set_kernel_threads(n: int) -> None:
    """内核 OpenMP 线程数. 非可用状态静默跳过."""
    if not kernel_available():
        return
    lib = load_kernel()
    lib.sn_set_threads(int(n))


if __name__ == "__main__":
    if not kernel_available():
        print(TOOLCHAIN_HINT)
        sys.exit(1)
    lib = load_scan_kernel(force_rebuild=True)
    print(f"编译并加载成功: {DLL}")