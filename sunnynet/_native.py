"""预编译共享库的定位与构建.

包内 sunnynet/_prebuilt/<平台标签>/ 存放构建期编好的产物。运行期优先加载它,
命中就不再要求本机有 C 编译器; 未命中或 CPU 指令集不达标时退回源码编译。

预编译库一律以 -mavx2 -mfma -march=x86-64-v3 编出, 只对 x86-64 且支持 AVX2 的
机器可用, 故按指令集判定而非仅按平台。
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

__all__ = [
    "PREBUILT", "platform_tag", "cpu_flags", "isa_ok",
    "locate", "build_prebuilt", "MODULES",
]

HERE = Path(__file__).resolve().parent
PREBUILT = HERE / "_prebuilt"

# 构建期需要产出预编译库的模块; 每个模块需有 _compile(gcc) 与 DLL 属性。
MODULES = (
    "sunnynet.runtime",
    "sunnynet.runtime.model",
    "sunnynet.runtime.rwkv7_c",
    "sunnynet.runtime.quant",
    "sunnynet.runtime.abi",
    "sunnynet.kernels",
)


def _machine() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def platform_tag() -> str:
    """对应 _prebuilt 下的子目录名, 如 win-x86_64 / linux-x86_64。"""
    if sys.platform == "win32":
        return f"win-{_machine()}"
    if sys.platform == "darwin":
        return f"macos-{_machine()}"
    return f"linux-{_machine()}"


def cpu_flags() -> set[str]:
    """本机 CPU 特性名集合 (小写)。"""
    if sys.platform != "win32":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8",
                                                  errors="replace")
        except OSError:
            return set()
        for line in text.splitlines():
            low = line.lower()
            if low.startswith("flags") or low.startswith("features"):
                return set(line.split(":", 1)[1].split())
        return set()
    try:
        import ctypes
        import ctypes.wintypes as wt

        k = ctypes.windll.kernel32
        k.IsProcessorFeaturePresent.argtypes = [wt.DWORD]
        k.IsProcessorFeaturePresent.restype = wt.BOOL
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40。Windows 无独立的 FMA 查询位,
        # 但 FMA3 与 AVX2 同期引入 (Haswell), 有 AVX2 即有 FMA。
        return {"avx2", "fma"} if k.IsProcessorFeaturePresent(40) else set()
    except Exception:
        return set()


_ISA_CACHE = None


def isa_ok() -> bool:
    """预编译库用 x86-64-v3 编出, 只有 AVX2+FMA 的 x86-64 机器能跑。"""
    global _ISA_CACHE
    if _ISA_CACHE is None:
        if _machine() != "x86_64":
            _ISA_CACHE = False
        else:
            f = cpu_flags()
            _ISA_CACHE = "avx2" in f and "fma" in f
    return _ISA_CACHE


def find(name: str) -> Path | None:
    p = PREBUILT / platform_tag() / name
    return p if p.is_file() else None


def locate(shlib_name: str) -> Path | None:
    """预编译库可加载则返回路径, 否则 None (调用方退回源码编译)。

    设 SUNNY_NO_PREBUILT=1 可强制走源码路径, 用于对照验证。
    """
    if os.environ.get("SUNNY_NO_PREBUILT") == "1" or not isa_ok():
        return None
    return find(shlib_name)


def build_prebuilt(out_dir: str | Path | None = None):
    """编译全部原生库并复制到 _prebuilt/<平台标签>/。

    返回 (产物列表, 错误说明)。当前 C 运行时用 -mavx2 编出, 只有 x86-64 能编;
    其他架构返回空列表并说明, 由调用方决定是否视为失败。
    """
    import importlib

    if _machine() != "x86_64":
        return [], f"{_machine()} 不是 x86-64, 该架构无预编译产物"
    if not isa_ok():
        return [], "构建机不支持 AVX2+FMA, 跳过预编译"

    out = Path(out_dir) if out_dir else PREBUILT / platform_tag()
    made: list[Path] = []
    for name in MODULES:
        mod = importlib.import_module(name)
        gcc = mod._find_gcc()
        if gcc is None:
            return made, f"未找到 C 编译器, 无法生成预编译库 ({name})"
        mod._compile(gcc)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / mod.DLL.name
        shutil.copy2(mod.DLL, dst)
        made.append(dst)
    return made, None