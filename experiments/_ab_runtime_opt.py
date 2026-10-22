"""优化前 / 优化后 C runtime 的严格 A/B: 数值一致性与速度。

优化前: 转置布局 (in,out) + 单线程 GEMV。
优化后: 行主序 (out,in) + 物理核 OpenMP。
同一权重 blob, 同一 token 序列。数值差异必须远低于 torch 对照的容差
(tests/test_runtime_model.py 用 1e-3), 否则说明布局翻错。
"""
from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "experiments/ckpts_100m/sunnynet_p100mw.pt"
BLOB = ROOT / "dist/release/sunnynet-p100mw.runtime.bin"
PROMPT = [1] + [100 + (i * 37) % 4000 for i in range(31)]
SIZE = BLOB.stat().st_size
# 优化前的库: 由 git show 出的源码编译, 路径用 SUNNY_OLD_DLL 传入
OLD_DLL = Path(os.environ.get("SUNNY_OLD_DLL", "/tmp/old_model.dll"))


def bind(dll_path):
    lib = ctypes.CDLL(str(dll_path))
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn_load.restype = V
    lib.sn_load.argtypes = [P, I, I, I, I, I, I, I, I]
    lib.sn_step.argtypes = [V, I, V]
    return lib, V, I


def main():
    from sunnynet.cli import load_model
    from sunnynet.runtime.model import SnModel, set_threads, recommended_threads

    if not OLD_DLL.exists():
        raise SystemExit(f"缺少对照库 {OLD_DLL}; 见 _bench 脚本头注释")
    _, cfg, _ = load_model(str(CKPT))
    d = cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg)
    args = (d["vocab_size"], d["dim"], d["inner"], d["rank"],
            d.get("taps", 3), d["layers"], d.get("group", 8), 1)

    # 新实现 (走 Python 层, 会自动按物理核并行)
    set_threads(recommended_threads())
    h_new = SnModel(BLOB, *args)
    h_new.reset()
    new = np.stack([np.asarray(h_new.step(t), dtype=np.float32) for t in PROMPT])

    # 旧实现 (直接绑定 /tmp/old_model.dll, 单线程)
    lib_old, V, I = bind(OLD_DLL)
    buf = (ctypes.c_float * d["vocab_size"])()
    h = lib_old.sn_load(str(BLOB).encode(), *[int(x) for x in args])
    if not h:
        raise SystemExit("旧库 sn_load 失败")
    lib_old.sn_reset.argtypes = [V]
    lib_old.sn_reset(h)
    old = []
    for t in PROMPT:
        lib_old.sn_step(h, int(t), ctypes.byref(buf))
        old.append(np.frombuffer(buf, dtype=np.float32).copy())
    old = np.stack(old)

    diff = float(np.abs(old - new).max())
    am_o, am_n = old.argmax(-1), new.argmax(-1)
    print(f"A/B 同一 blob  n={len(PROMPT)}")
    print(f"  logits max|diff| = {diff:.3e}")
    print(f"  argmax 一致      = {int((am_o == am_n).sum())}/{len(PROMPT)}")
    print(f"  逐位相同         = {bool((old == new).all())}")

    # 速度对战
    reps = 5
    t_old, t_new = [], []
    for r in range(reps):
        if r % 2 == 0:
            lib_old.sn_reset(h)
            t0 = time.perf_counter()
            for t in PROMPT:
                lib_old.sn_step(h, int(t), ctypes.byref(buf))
            t_old.append((time.perf_counter() - t0) / len(PROMPT))

            h_new.reset()
            t0 = time.perf_counter()
            for t in PROMPT:
                h_new.step(t)
            t_new.append((time.perf_counter() - t0) / len(PROMPT))
        else:
            h_new.reset()
            t0 = time.perf_counter()
            for t in PROMPT:
                h_new.step(t)
            t_new.append((time.perf_counter() - t0) / len(PROMPT))

            lib_old.sn_reset(h)
            t0 = time.perf_counter()
            for t in PROMPT:
                lib_old.sn_step(h, int(t), ctypes.byref(buf))
            t_old.append((time.perf_counter() - t0) / len(PROMPT))

    mo, mn = statistics.median(t_old) * 1e3, statistics.median(t_new) * 1e3
    wins = sum(a < b for a, b in zip(t_new, t_old))
    print(f"\n单步 ms/token ({reps} 轮交错 head-to-head)")
    print(f"  优化前 (转置+单线程) 中位 {mo:7.2f}  min {min(t_old)*1e3:7.2f}"
          f"   带宽 {SIZE/(mo/1e3)/1e9:6.2f} GB/s")
    print(f"  优化后 (行主序+并行) 中位 {mn:7.2f}  min {min(t_new)*1e3:7.2f}"
          f"   带宽 {SIZE/(mn/1e3)/1e9:6.2f} GB/s")
    print(f"  优化后胜 {wins}/{reps}   加速 {mo/mn:.2f}x")
    print(f"  线程数 {recommended_threads()} (物理核)")


if __name__ == "__main__":
    main()