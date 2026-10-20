"""GPU 验证: 同一 SunnyNet 在 CPU 与 CUDA 上的前向/训练步耗时与显存.

用法: python -X utf8 experiments/_gpu_probe.py <device: cpu|cuda> <preset> <T> <B> <steps>
输出单行 JSON; 无 CUDA 时如实报错.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
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
    "small": lambda v: SunnyConfig.small(v),
    "proto20m": lambda v: SunnyConfig.proto20m(v),
    "proto20mw": lambda v: SunnyConfig(vocab_size=v, dim=640, inner=1280, rank=384,
                                       layers=6, h_max=8192.0),
    "p100m": lambda v: SunnyConfig(vocab_size=v, dim=1024, inner=1792, rank=640,
                                   layers=14, h_max=16384.0),
    "p100mw": lambda v: SunnyConfig(vocab_size=v, dim=1408, inner=1920, rank=1024,
                                    layers=8, h_max=16384.0),
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
    dev = sys.argv[1]
    preset = sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    info = {"device": dev, "preset": preset, "T": T, "B": B,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if dev == "cuda" and not torch.cuda.is_available():
        info["error"] = "CUDA 不可用"
        print("RESULT " + json.dumps(info))
        return

    device = torch.device("cuda" if dev == "cuda" else "cpu")
    if dev == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
        torch.backends.cudnn.benchmark = False

    vocab = 8192
    cfg = PRESETS[preset](vocab)
    t0 = time.perf_counter()
    try:
        m = SunnyNet(cfg).to(device).train()
        if dev == "cuda":
            torch.cuda.synchronize()
    except (RuntimeError, MemoryError) as e:
        info["build_error"] = str(e)[:300]
        if dev == "cuda":
            info["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            info["cuda_free_mb"] = round(torch.cuda.mem_get_info()[0] / 2 ** 20)
        print("RESULT " + json.dumps(info, ensure_ascii=False))
        return
    info["build_s"] = round(time.perf_counter() - t0, 2)
    info["stack"] = m.num_params(non_embedding=True)
    info["total"] = m.num_params()

    x = torch.randint(0, vocab, (B, T), device=device)
    y = torch.randint(0, vocab, (B, T), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def fwd():
        return m(x)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    try:
        for _ in range(2):
            fwd()
        if dev == "cuda":
            torch.cuda.synchronize()
        ft = []
        for _ in range(5):
            t0 = time.perf_counter()
            fwd()
            if dev == "cuda":
                torch.cuda.synchronize()
            ft.append(time.perf_counter() - t0)
        info["fwd_ms_median"] = round(statistics.median(ft) * 1000, 3)
        info["fwd_ms_per_token"] = round(statistics.median(ft) * 1000 / (B * T), 5)
    except (RuntimeError, MemoryError) as e:
        info["fwd_error"] = str(e)[:300]
        if dev == "cuda":
            info["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            info["cuda_free_mb"] = round(torch.cuda.mem_get_info()[0] / 2 ** 20)
        print("RESULT " + json.dumps(info, ensure_ascii=False))
        return

    try:
        for _ in range(steps):
            step()
        if dev == "cuda":
            torch.cuda.synchronize()
        tt = []
        for _ in range(3):
            t0 = time.perf_counter()
            step()
            if dev == "cuda":
                torch.cuda.synchronize()
            tt.append(time.perf_counter() - t0)
        info["step_ms_median"] = round(statistics.median(tt) * 1000, 3)
        info["step_ms_per_token"] = round(statistics.median(tt) * 1000 / (B * T), 5)
    except RuntimeError as e:
        info["step_error"] = str(e)[:200]

    if dev == "cuda":
        info["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        info["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
        info["cuda_free_mb"] = round(torch.cuda.mem_get_info()[0] / 2 ** 20)
    else:
        _, p = rss_mb()
        info["peak_rss_mb"] = round(p)

    print("RESULT " + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()