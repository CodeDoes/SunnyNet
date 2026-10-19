"""SunnyNet: 原创 CPU 通用序列神经网络架构."""
from .model import SunnyConfig, SunnyNet
from .ops import CenteredScaleNorm, logspace_scan, poly_sigmoid, softwedge

__version__ = "0.1.0"
__all__ = ["SunnyConfig", "SunnyNet", "CenteredScaleNorm", "logspace_scan", "poly_sigmoid", "softwedge"]
