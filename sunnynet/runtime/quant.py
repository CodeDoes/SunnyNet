"""INT4 量化原语的 ctypes 绑定 (不依赖 torch)."""
from __future__ import annotations

import ctypes
from pathlib import Path

from . import BUILD, HERE, _find_gcc

__all__ = ["quant_available", "load_quant", "I4_BLOCK"]

I4_BLOCK = 64
DLL = BUILD / "sunnynet_quant.dll"


def quant_available() -> bool:
    return _find_gcc() is not None


def _compile(gcc: str) -> None:
    import subprocess
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = [gcc, "-O3", "-mavx2", "-mfma", "-march=x86-64-v3",
           "-funroll-loops", "-fopenmp", "-static", "-shared", "-std=c11",
           str(HERE / "quant.c"), "-o", str(DLL)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"quant 编译失败: {(r.stderr or r.stdout)[-800:]}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    t = DLL.stat().st_mtime
    return any(s.stat().st_mtime > t for s in (HERE / "quant.c", HERE / "quant.h"))


_lib = None


def load_quant(force_rebuild: bool = False):
    global _lib
    if _lib is not None and not force_rebuild:
        return _lib
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器 (设 SUNNY_W64)")
    if force_rebuild or _needs_rebuild():
        _compile(gcc)
    lib = ctypes.CDLL(str(DLL))
    V, I = ctypes.c_void_p, ctypes.c_int
    lib.sn_quant_i4.argtypes = [V, V, V, I, I, I]
    lib.sn_dequant_i4.argtypes = [V, V, V, I, I, I]
    lib.sn_gemv_i4.argtypes = [V, V, V, V, I, I, I]
    lib.sn_gemv_f32.argtypes = [V, V, V, I, I]
    _lib = lib
    return lib


def _ptr(arr):
    return ctypes.c_void_p(arr.ctypes.data)


def quant_i4(x, block: int = I4_BLOCK):
    """x: (rows, N) fp32 -> (q uint8 (rows, N/2), scale fp32 (rows, N/block))"""
    import numpy as np
    lib = load_quant()
    rows, N = x.shape
    x = np.ascontiguousarray(x, dtype=np.float32)
    nb = N // block
    q = np.empty((rows, N // 2), dtype=np.uint8)
    scale = np.empty((rows, nb), dtype=np.float32)
    lib.sn_quant_i4(_ptr(x), _ptr(q), _ptr(scale), rows, N, block)
    return q, scale


def dequant_i4(q, scale, N: int, block: int = I4_BLOCK):
    import numpy as np
    lib = load_quant()
    rows = q.shape[0]
    q = np.ascontiguousarray(q, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.float32)
    x = np.empty((rows, N), dtype=np.float32)
    lib.sn_dequant_i4(_ptr(q), _ptr(scale), _ptr(x), rows, N, block)
    return x


def gemv_i4(x, q, scale, block: int = I4_BLOCK):
    """y = W x; W 由 (q, scale) 以 (out, in) 布局给出."""
    import numpy as np
    lib = load_quant()
    x = np.ascontiguousarray(x, dtype=np.float32)
    in_dim = x.shape[0]
    out = q.shape[0]
    q = np.ascontiguousarray(q, dtype=np.uint8)
    scale = np.ascontiguousarray(scale, dtype=np.float32)
    y = np.empty(out, dtype=np.float32)
    lib.sn_gemv_i4(_ptr(y), _ptr(x), _ptr(q), _ptr(scale), in_dim, out, block)
    return y


def gemv_f32(x, W):
    import numpy as np
    lib = load_quant()
    x = np.ascontiguousarray(x, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    out, in_dim = W.shape
    y = np.empty(out, dtype=np.float32)
    lib.sn_gemv_f32(_ptr(y), _ptr(x), _ptr(W), in_dim, out)
    return y