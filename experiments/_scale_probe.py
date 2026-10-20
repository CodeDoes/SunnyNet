"""规模化内存可行性探测: 实测各预设的峰值 RSS 与训练步耗时.

用法: python -X utf8 experiments/_scale_probe.py <preset> <T> <B> <steps> [threads]
     preset: small | proto20m | p100m | scale05b
输出: 单行 JSON; 进程被 SIGTERM (OOM) 时由外层捕获
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

PRESETS = {
    "small": lambda v: SunnyConfig.small(v),
    "proto20m": lambda v: SunnyConfig.proto20m(v),
    "p100m": lambda v: SunnyConfig(vocab_size=v, dim=1024, inner=1792, rank=640,
                                   layers=14, h_max=16384.0),
    "scale05b": lambda v: SunnyConfig.scale05b(v),
}


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def rss_mb():
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wt.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2 ** 20, c.PeakWorkingSetSize / 2 ** 20


def main():
    preset = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    torch.set_num_threads(threads)
    vocab = 8192
    cfg = PRESETS[preset](vocab)
    t0 = time.perf_counter()
    m = SunnyNet(cfg).train()
    built = time.perf_counter() - t0
    r0, p0 = rss_mb()

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    per = (time.perf_counter() - t0) / steps
    r1, p1 = rss_mb()

    print("RESULT " + json.dumps({
        "preset": preset, "T": T, "B": B, "steps": steps,
        "stack": m.num_params(non_embedding=True), "total": m.num_params(),
        "build_s": round(built, 1),
        "rss_after_build_mb": round(r0), "rss_after_step_mb": round(r1),
        "peak_rss_mb": round(p1),
        "step_s": round(per, 3), "ms_per_token": round(per * 1000 / (B * T), 4),
    }))


if __name__ == "__main__":
    main()