"""解码态: C 整模型后端 (扫线程数) vs torch eager, 同规格 §23.3 的 sunny 格.

口径: 每格独立子进程, torch 固定 1 线程, 先走满 T 步建状态再测单步, 多轮取中位。
输出 C 后端在不同线程数下的单步耗时, 定位小模型下 OpenMP 线程数的收益拐点。
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V = 8192
SPEC = (80, 160, 244, 4)
CKPT = ROOT / "experiments/ckpts_decode/d80.pt"
BLOB = ROOT / "experiments/ckpts_decode/d80.runtime.bin"

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
T, kind, thr, reps = {T}, "{kind}", {thr}, {reps}
torch.set_num_threads(1); torch.manual_seed(1234)
from sunnynet.cli import load_model
m, cfg, _ = load_model(r"{ckpt}")
m.eval()
if kind == "c":
    from sunnynet.runtime.model import SnModel, set_threads
    rt = SnModel(r"{blob}", cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                 cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings))
    set_threads(thr)
idx = torch.randint(0, cfg.vocab_size, (1, T))
one = int(torch.randint(0, cfg.vocab_size, (1,)).item())
if kind == "c":
    rt.prefill(idx[0].tolist()); rt.step(one)
    fn = lambda: rt.step(one)
else:
    s = m.init_state(1)
    with torch.no_grad():
        for t in range(T):
            _, s = m.forward_step(idx[:, t], s)
        o = torch.tensor([one]); m.forward_step(o, s)
    fn = lambda: m.forward_step(o, s)
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
print("CELL " + json.dumps({{"ms": st.median(ts) * 1000}}))
'''


def reps_for(T):
    return 64 if T <= 1024 else (32 if T <= 2048 else 16)


def run_child(T, kind, thr, reps):
    code = CELL.format(root=str(ROOT), T=T, kind=kind, thr=thr, reps=reps,
                       ckpt=CKPT, blob=BLOB)
    r = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), env=dict(os.environ))
    for ln in r.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])["ms"]
    raise RuntimeError((r.stderr or r.stdout or "")[-1000:])


def main():
    sys.path.insert(0, str(ROOT))
    if not CKPT.exists() or not BLOB.exists():
        raise SystemExit("先跑 experiments/_decode_cbackend.py 生成 ckpt/blob")
    Ts = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                           else ["256", "1024", "2048"])]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    kinds = [("torch", 1), ("c", 1), ("c", 2), ("c", 4), ("c", 6)]
    acc = {(k, thr, T): [] for k, thr in kinds for T in Ts}
    for rd in range(rounds):
        for T in Ts:
            for k, thr in kinds:
                v = run_child(T, k, thr, reps_for(T))
                acc[(k, thr, T)].append(v)
                print(f"  r{rd+1} {k}/thr{thr} T={T:<5d} {v:.4f} ms")
    out = {"Ts": Ts, "rounds": rounds, "table": {}}
    print("\n=== 单步解码 ms 中位 (spec d80/i160/r244/L4, 与 §23.3 sunny 格一致) ===")
    print("group".ljust(14) + "".join(f"T={T}".ljust(11) for T in Ts))
    for k, thr in kinds:
        key = f"{k}/thr{thr}"
        row = [st.median(acc[(k, thr, T)]) for T in Ts]
        out["table"][key] = row
        print(f"  {key:12s}" + "".join(f"{v:.4f}".ljust(11) for v in row))
    # 相对 torch 的比值
    base = out["table"]["torch/thr1"]
    print("\n=== 相对 torch 加速 (>1 更快) ===")
    for k, thr in kinds:
        if k == "torch":
            continue
        key = f"{k}/thr{thr}"
        rr = [t / v for t, v in zip(base, out["table"][key])]
        out.setdefault("speedup", {})[key] = rr
        print(f"  {key:12s}" + "".join(f"{v:.3f}".ljust(11) for v in rr))
    p = ROOT / "experiments/results/_decode_cthreads.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())