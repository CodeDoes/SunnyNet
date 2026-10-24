"""SunnyNet: 原创 CPU 通用序列神经网络架构."""
import importlib

__version__ = "0.3.0"
__all__ = ["SunnyConfig", "SunnyNet", "CenteredScaleNorm", "logspace_scan",
           "poly_sigmoid", "softwedge"]

# 延迟导入 torch: kernels/runtime 子包不依赖它, 顶层 import 会让二者无法单独导入。
_LAZY = {
    "SunnyConfig": "sunnynet.model",
    "SunnyNet": "sunnynet.model",
    "CenteredScaleNorm": "sunnynet.ops",
    "logspace_scan": "sunnynet.ops",
    "poly_sigmoid": "sunnynet.ops",
    "softwedge": "sunnynet.ops",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    val = getattr(importlib.import_module(mod), name)
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(__all__) | set(globals()))