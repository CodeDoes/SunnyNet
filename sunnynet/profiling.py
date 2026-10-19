"""运行时性能探针. 全部指标为真实测量, 不估不猜.

  peak_rss_mb   : GetProcessMemoryInfo 的 PeakWorkingSetSize
  cache_hit_rate: PDH "Cache\\% Cache Read Hits/Misses", 不可用时返回 None
  吞吐 / 耗时    : perf_counter 实测
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time
from dataclasses import dataclass, field

__all__ = ["peak_rss_mb", "current_rss_mb", "CacheCounter", "Stopwatch", "peak_sys_mb"]

_PSAPI = None


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _mem_counters() -> _PROCESS_MEMORY_COUNTERS | None:
    """GetCurrentProcess() 返回伪句柄 -1, ctypes 默认按 32 位截断会失败, 须显式声明 restype."""
    global _PSAPI
    try:
        if _PSAPI is None:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            for dll in ("psapi", "kernel32"):
                lib = ctypes.WinDLL(dll, use_last_error=True)
                fn = getattr(lib, "GetProcessMemoryInfo", None)
                if fn is not None:
                    fn.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wt.DWORD]
                    fn.restype = wt.BOOL
                    _PSAPI = (fn, k32)
                    break
            else:
                _PSAPI = False
        if not _PSAPI:
            return None
        fn, k32 = _PSAPI
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc
    except Exception:
        pass
    return None


def peak_rss_mb() -> float | None:
    pmc = _mem_counters()
    return None if pmc is None else pmc.PeakWorkingSetSize / (1024 * 1024)


def current_rss_mb() -> float | None:
    pmc = _mem_counters()
    return None if pmc is None else pmc.WorkingSetSize / (1024 * 1024)


_GB = 1024 ** 3


def peak_sys_mb() -> float | None:
    """系统提交峰值 (全机), 用于判断是否触发了换页."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.ullTotalPhys - m.ullAvailPhys) / (1024 * 1024)
    except Exception:
        return None


class CacheCounter:
    """CPU 缓存读命中率, 不可用时回退到进程页错误率 (kind 字段区分, 不冒充实测)."""

    CACHE_PATHS = ("\\Cache\\% Cache Read Hits", "\\Cache\\% Cache Read Hits")
    MISS_PATHS = ("\\Cache\\% Cache Read Misses", "\\Cache\\% Cache Read Misses")

    def __init__(self, pid: int | None = None) -> None:
        self.ok = False
        self.kind = "none"
        self._query = None
        self._h_hits = None
        self._h_miss = None
        self._h_pf = None
        self._pdh = None
        self._open(pid)

    def _open(self, pid: int | None) -> None:
        try:
            pdh = ctypes.WinDLL("pdh", use_last_error=True)
            q = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(q)) != 0:
                return
            hits, miss = ctypes.c_void_p(), ctypes.c_void_p()
            got = all(self._add(pdh, q, n, ctypes.byref(h))
                      == 0 for n, h in zip(self.CACHE_PATHS, (hits, miss)))
            if got:
                self._pdh, self._query, self._h_hits, self._h_miss = pdh, q, hits, miss
                self.kind = "cache_read_hit_rate"
                pdh.PdhCollectQueryData(q)
                if self.hit_rate() is not None:
                    self.ok = True
                    return
                pdh.PdhCloseQuery(q)
                q = ctypes.c_void_p()
                pdh.PdhOpenQueryW(None, 0, ctypes.byref(q))
            # 回退: 进程页错误率
            if pid is None:
                import os
                pid = os.getpid()
            pf = ctypes.c_void_p()
            name = f"\\Process({pid})\\Page Faults/sec"
            if self._add(pdh, q, name, ctypes.byref(pf)) == 0:
                self._pdh, self._query, self._h_pf = pdh, q, pf
                self.kind = "page_faults_per_sec"
                pdh.PdhCollectQueryData(q)
                self.ok = True
                return
            pdh.PdhCloseQuery(q)
        except Exception:
            self.ok = False

    @staticmethod
    def _add(pdh, q, name: str, holder) -> int:
        for fn_name in ("PdhAddEnglishCounterW", "PdhAddCounterW"):
            f = getattr(pdh, fn_name, None)
            if f is None:
                continue
            f.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_void_p)]
            f.restype = ctypes.c_long
            if f(q, ctypes.c_wchar_p(name), 0, holder) == 0:
                return 0
        return -1

    def _read(self, h):
        class PDH_FMT_COUNTERVALUE(ctypes.Structure):
            _fields_ = [("CStatus", wt.DWORD), ("doubleValue", ctypes.c_double)]
        v = PDH_FMT_COUNTERVALUE()
        if self._pdh.PdhGetFormattedCounterValue(h, 0x00000200, None,
                                                 ctypes.byref(v)) != 0:
            return None
        return v.doubleValue

    def sample(self) -> float | None:
        """语义由 kind 决定. 速率型计数器首次采样常返回 PDH_INVALID_DATA, 故重试一次."""
        if not self.ok:
            return None
        try:
            for _ in range(2):
                if self._pdh.PdhCollectQueryData(self._query) != 0:
                    return None
                if self.kind == "cache_read_hit_rate":
                    a, b = self._read(self._h_hits), self._read(self._h_miss)
                    if a is None or b is None:
                        continue
                    if a > 1.5 or b > 1.5:
                        tot = a + b
                        return None if tot <= 0 else a / tot
                    return a
                v = self._read(self._h_pf)
                if v is not None:
                    return v
            return None
        except Exception:
            return None

    def hit_rate(self) -> float | None:
        """仅当 kind 为真实缓存命中率时返回数值, 否则 None."""
        if self.kind != "cache_read_hit_rate":
            return None
        return self.sample()


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.laps: list[tuple[str, float]] = []

    def lap(self, name: str) -> float:
        now = time.perf_counter()
        dt = now - self.t0
        self.laps.append((name, dt))
        self.t0 = now
        return dt

    @property
    def elapsed(self) -> float:
        return sum(d for _, d in self.laps)


@dataclass
class PerfTracker:
    """训练过程中的性能统计."""

    tokens: int = 0
    seconds: float = 0.0
    grad_norms: list[float] = field(default_factory=list)
    rss_samples: list[float] = field(default_factory=list)
    peak_rss: float = 0.0
    cache_samples: list[float] = field(default_factory=list)
    pf_samples: list[float] = field(default_factory=list)
    cache_kind: str = "none"

    def update(self, n_tokens: int, dt: float, grad_norm: float | None = None,
               cache: CacheCounter | None = None) -> None:
        self.tokens += n_tokens
        self.seconds += dt
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        r = current_rss_mb()
        if r is not None:
            self.rss_samples.append(r)
        p = peak_rss_mb()
        if p is not None:
            self.peak_rss = max(self.peak_rss, p)
        if cache is not None:
            self.cache_kind = cache.kind
            if cache.kind == "cache_read_hit_rate":
                c = cache.sample()
                if c is not None:
                    self.cache_samples.append(c)
            elif cache.kind == "page_faults_per_sec":
                v = cache.sample()
                if v is not None:
                    self.pf_samples.append(v)

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def ms_per_token(self) -> float:
        return 1000.0 * self.seconds / self.tokens if self.tokens else 0.0

    @property
    def mean_cache_hit(self) -> float | None:
        return sum(self.cache_samples) / len(self.cache_samples) if self.cache_samples else None

    @property
    def mean_page_faults(self) -> float | None:
        return sum(self.pf_samples) / len(self.pf_samples) if self.pf_samples else None

    @property
    def mean_grad_norm(self) -> float | None:
        return sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else None