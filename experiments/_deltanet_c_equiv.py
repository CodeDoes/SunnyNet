"""DeltaNet 整模型 C 内核 vs numpy 参考: 逐 token 一致性与速度.

C 侧 blob 由 experiments/_export_deltanet.py 生成 (体积大, 不入库)。
参考路径 sunnynet.runtime.deltanet.DeltaNet 是独立实现 (numpy/einsum),
两者不一致即说明 C 侧写错。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MDIR = os.environ.get("SUNNY_DELTANET", str(ROOT / "models" / "deltanet-pure"))
DLL = ROOT / "sunnynet" / "runtime" / "_build" / "sunnynet_deltanet.dll"


def bind():
    lib = ctypes.CDLL(str(DLL))
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn_dn_load.restype = V
    lib.sn_dn_load.argtypes = [P]
    lib.sn_dn_vocab.restype = I
    lib.sn_dn_vocab.argtypes = [V]
    lib.sn_dn_ready.restype = I
    lib.sn_dn_ready.argtypes = [V]
    lib.sn_dn_reset.argtypes = [V]
    lib.sn_dn_step.argtypes = [V, I, V]
    lib.sn_dn_free.argtypes = [V]
    return lib


def main():
    from sunnynet.runtime.deltanet import DeltaNet

    lib = bind()
    blob = Path(MDIR) / "deltanet.c.bin"
    h = lib.sn_dn_load(str(blob).encode())
    if not h or not lib.sn_dn_ready(h):
        print(f"加载失败: {blob}")
        return
    print(f"C 侧 vocab={lib.sn_dn_vocab(h)}")

    m = DeltaNet(MDIR)
    st = m.reset(m.init_state())
    ids = [1, 415, 5565, 302, 4843, 349] + list(range(10, 40))
    logits = (ctypes.c_float * m.vocab)()

    dmax, am = 0.0, 0
    for t in ids:
        ref = m.step(int(t), st)
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))
        c = np.frombuffer(logits, dtype=np.float32)
        dmax = max(dmax, float(np.abs(c - ref).max()))
        am += int(c.argmax() == ref.argmax())
    print(f"1) 逐 token ({len(ids)} 步) logits max|diff| = {dmax:.3e}  "
          f"argmax {am}/{len(ids)}")

    # 分段续算: 状态可迁移
    lib.sn_dn_reset(h)
    a = []
    for t in ids:
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))
        a.append(np.frombuffer(logits, dtype=np.float32).copy())
    lib.sn_dn_reset(h)
    b = []
    for t in ids:
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))
        b.append(np.frombuffer(logits, dtype=np.float32).copy())
    print(f"2) 重复运行 bit 稳定 max|diff| = "
          f"{max(float(np.abs(x - y).max()) for x, y in zip(a, b)):.3e}")

    # 分段: 传满一段后 reset, C 侧状态清零确实生效
    lib.sn_dn_reset(h)
    for t in ids[:8]:
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))
    lib.sn_dn_reset(h)
    seg = []
    for t in ids:
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))
        seg.append(np.frombuffer(logits, dtype=np.float32).copy())
    print(f"3) reset 后从头算与首次一致 max|diff| = "
          f"{max(float(np.abs(x - y).max()) for x, y in zip(a, seg)):.3e}")

    # 速度: 同进程交替互测
    warm = ids[:20]
    st2 = m.reset(m.init_state())
    for t in warm:
        m.step(int(t), st2)
    lib.sn_dn_reset(h)
    for t in warm:
        lib.sn_dn_step(h, int(t), ctypes.byref(logits))

    def timeit(fn, reps=30, warm=5):
        for i in range(warm):
            fn(i)
        ts = []
        for i in range(reps):
            t0 = time.perf_counter()
            fn(i)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2]

    nc = timeit(lambda i: m.step(int(ids[i % len(ids)]), st2))
    cc = timeit(lambda i: lib.sn_dn_step(h, int(ids[i % len(ids)]),
                                         ctypes.byref(logits)))
    print(f"4) numpy {nc*1e3:.2f} ms/token | C {cc*1e3:.2f} ms/token | "
          f"{nc/cc:.2f}x")
    lib.sn_dn_free(h)


if __name__ == "__main__":
    main()