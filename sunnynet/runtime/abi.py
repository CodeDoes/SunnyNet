"""稳定 C ABI 的 ctypes 绑定 (不依赖 torch).

对应头文件 sn_abi.h; 符号以 sn_abi_ 前缀隔离于内部内核。
库把 sn_abi.c 与内部实现 (recur.c / rwkv7.c) 一起编译, 内部符号不对外承诺。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

from . import BUILD, HERE, _find_gcc, compile_flags, shlib_name

__all__ = ["abi_available", "load_abi", "ABI_VERSION", "CAPS", "STATUS_OK"]

DLL = BUILD / shlib_name("sunnynet_abi")

# 与 sn_abi.h 保持一致; 改动需同步两侧。
ABI_MAJOR, ABI_MINOR, ABI_PATCH = 1, 0, 0
ABI_VERSION = (ABI_MAJOR << 16) | (ABI_MINOR << 8) | ABI_PATCH

CAP_RECUR_SCAN = 1 << 0
CAP_RECUR_STEP = 1 << 1
CAP_QUANT_I8 = 1 << 2
CAP_RWKV7 = 1 << 3
CAP_THREADS = 1 << 4
CAPS = ("recur_scan", "recur_step", "quant_i8", "rwkv7", "threads")

STATUS_OK = 0
_STATUS_MSG = {
    0: "OK",
    -1: "E_INVAL 参数非法",
    -2: "E_NOMEM 分配失败",
    -3: "E_IO 读写失败",
    -4: "E_FORMAT 魔数/版本不匹配",
    -5: "E_UNSUPPORTED 该构建不支持",
}

SOURCES = ["sn_abi.c", "recur.c", "rwkv7.c", "quant.c"]


def abi_available() -> bool:
    from .. import _native
    return _find_gcc() is not None or _native.locate(DLL.name) is not None


def _compile(gcc: str) -> None:
    import subprocess
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = [gcc] + compile_flags()
    cmd += [str(HERE / s) for s in SOURCES]
    cmd += ["-o", str(DLL)]
    # 编译器在 Windows 下按系统代码页输出路径, 非 UTF-8; 报错解码需容错
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ABI 库编译失败: {err[-900:]}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    t = DLL.stat().st_mtime
    return any((HERE / s).stat().st_mtime > t for s in SOURCES + ["sn_abi.h"])


_lib = None


def _bind(lib):
    V, I, P = ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p
    F = ctypes.POINTER(ctypes.c_float)
    lib.sn_abi_version.restype = I
    lib.sn_abi_caps.restype = ctypes.c_uint32
    lib.sn_abi_check.argtypes = [I]
    lib.sn_abi_check.restype = I
    lib.sn_abi_version_string.argtypes = [P, I]
    lib.sn_abi_version_string.restype = I
    lib.sn_abi_backend_string.argtypes = [P, I]
    lib.sn_abi_backend_string.restype = I
    lib.sn_abi_set_threads.argtypes = [I]
    lib.sn_abi_set_threads.restype = I
    lib.sn_abi_get_threads.restype = I
    lib.sn_abi_recur_step.argtypes = [F, F, F, I, I, I]
    lib.sn_abi_recur_step.restype = I
    lib.sn_abi_recur_scan.argtypes = [F, F, F, F, F, I, I, I, I]
    lib.sn_abi_recur_scan.restype = I
    lib.sn_abi_quant_i8.argtypes = [F, V, F, I, I, I]
    lib.sn_abi_quant_i8.restype = I
    lib.sn_abi_dequant_i8.argtypes = [V, F, F, I, I, I]
    lib.sn_abi_dequant_i8.restype = I
    lib.sn_abi_rwkv7_load.argtypes = [P]
    lib.sn_abi_rwkv7_load.restype = V
    lib.sn_abi_rwkv7_step.argtypes = [V, I, F]
    lib.sn_abi_rwkv7_step.restype = I
    lib.sn_abi_rwkv7_prefill.argtypes = [V, V, I, F]
    lib.sn_abi_rwkv7_prefill.restype = I
    lib.sn_abi_rwkv7_reset.argtypes = [V]
    lib.sn_abi_rwkv7_reset.restype = I
    lib.sn_abi_rwkv7_free.argtypes = [V]
    lib.sn_abi_rwkv7_vocab.argtypes = [V]
    lib.sn_abi_rwkv7_vocab.restype = I
    return lib


def load_abi(force_rebuild: bool = False):
    global _lib
    if _lib is not None and not force_rebuild:
        return _lib
    if not force_rebuild:
        from .. import _native
        pre = _native.locate(DLL.name)
        if pre is not None:
            _lib = _bind(ctypes.CDLL(str(pre)))
            return _lib
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器 (设 SUNNY_W64 或把 gcc 加入 PATH)")
    if force_rebuild or _needs_rebuild():
        _compile(gcc)
    _lib = _bind(ctypes.CDLL(str(DLL)))
    return _lib


def status_str(code: int) -> str:
    return _STATUS_MSG.get(int(code), f"未知错误码 {code}")


# --------------------------------------------------------------- 便捷封装
def version() -> int:
    return int(load_abi().sn_abi_version())


def caps() -> int:
    return int(load_abi().sn_abi_caps())


def cap_names() -> list[str]:
    m = caps()
    return [n for i, n in enumerate(CAPS) if m & (1 << i)]


def check(caller_major: int = ABI_MAJOR) -> int:
    return int(load_abi().sn_abi_check(int(caller_major)))


def version_string() -> str:
    lib = load_abi()
    n = lib.sn_abi_version_string(None, 0)
    buf = ctypes.create_string_buffer(n)
    lib.sn_abi_version_string(buf, n)
    return buf.value.decode()


def backend_string() -> str:
    lib = load_abi()
    n = lib.sn_abi_backend_string(None, 0)
    buf = ctypes.create_string_buffer(n)
    lib.sn_abi_backend_string(buf, n)
    return buf.value.decode()


class Rwkv7ABI:
    """稳定 ABI 的 RWKV-7 句柄封装."""

    def __init__(self, blob_path):
        self._lib = load_abi()
        self._h = self._lib.sn_abi_rwkv7_load(str(blob_path).encode())
        if not self._h:
            raise RuntimeError(f"sn_abi_rwkv7_load 失败: {blob_path}")
        self.vocab = int(self._lib.sn_abi_rwkv7_vocab(self._h))

    def reset(self):
        return int(self._lib.sn_abi_rwkv7_reset(self._h))

    def step(self, token: int) -> list[float]:
        buf = (ctypes.c_float * self.vocab)()
        rc = int(self._lib.sn_abi_rwkv7_step(self._h, int(token), buf))
        if rc != STATUS_OK:
            raise RuntimeError(status_str(rc))
        return list(buf)

    def prefill(self, tokens) -> list[float]:
        toks = list(tokens)
        ibuf = (ctypes.c_int32 * len(toks))(*toks)
        buf = (ctypes.c_float * self.vocab)()
        rc = int(self._lib.sn_abi_rwkv7_prefill(self._h, ibuf, len(toks), buf))
        if rc != STATUS_OK:
            raise RuntimeError(status_str(rc))
        return list(buf)

    def close(self):
        if getattr(self, "_h", None):
            self._lib.sn_abi_rwkv7_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass