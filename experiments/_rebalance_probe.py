"""单格: 构建 preset 模型, 测 batch=8 训练步耗时与峰值 RSS.

用法: python -X utf8 experiments/_rebalance_probe.py <preset> <T>
每格需独立子进程调用 (测量纪律).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

PRESETS = {
    "proto20m": lambda v: SunnyConfig.proto20m(v),
    "proto20mw": lambda v: SunnyConfig(vocab_size=v, dim=640, inner=1280, rank=384,
                                       layers=6, h_max=8192.0),
    "p100m": lambda v: SunnyConfig(vocab_size=v, dim=1024, inner=1792, rank=640,
                                   layers=14, h_max=16384.0),
    "p100mw": lambda v: SunnyConfig(vocab_size=v, dim=1408, inner=1920, rank=1024,
                                    layers=8, h_max=16384.0),
    "p100mw2": lambda v: SunnyConfig(vocab_size=v, dim=1280, inner=2048, rank=640,
                                     layers=10, h_max=16384.0),
}


def _peak_rss_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                           ctypes.POINTER(PMC), wintypes.DWORD]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / (1024.0 * 1024.0)


def main():
    preset = sys.argv[1]
    T = int(sys.argv[2])
    torch.set_num_threads(8)
    torch.manual_seed(0)
    V = 8192
    cfg = PRESETS[preset](V)
    m = SunnyNet(cfg).train()
    x = torch.randint(0, V, (8, T))
    y = torch.randint(0, V, (8, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = statistics.median(ts)
    rss = _peak_rss_mb()
    print(json.dumps({
        "preset": preset, "T": T, "layers": cfg.layers, "dim": cfg.dim,
        "stack": m.num_params(non_embedding=True),
        "step_s": round(med, 4),
        "ms_per_tok": round(med * 1000 / (8 * T), 4),
        "rss_mb": round(rss, 1),
        "iqr_ratio": round((ts[-1] - ts[0]) / med, 3),
    }))


if __name__ == "__main__":
    main()