"""端到端解码: torch 路径 vs 融合内核路径, 逐格独立子进程.

口径: 同一 ckpt / prompt 长度 / 线程数; 只切换递推算子实现
(SUNNY_KERNEL=0 走 torch 向量化扫描, =1 走 AVX2 融合内核);
先 prefill 走满 T 建状态, 再测 30 次单步的每步中位耗时; 5 轮取中位。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "results" / "_runtime_e2e.json"
PY = sys.executable
CKPT = "experiments/ckpts_ood/sunnynet.pt"


def child(kernel, threads, T, reps):
    import torch
    st = Path(ROOT).resolve()
    if kernel:
        from sunnynet.kernels.core_auto import set_kernel_active
        set_kernel_active(True, threads=threads)
    from sunnynet.cli import load_model
    model, cfg, _ = load_model(str(st / CKPT))
    torch.set_num_threads(threads)
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        lg, s = model(ids, return_state=True)
    cur = lg[0, -1].argmax().view(1)

    state = {"s": s, "cur": cur}

    def run():
        with torch.no_grad():
            for _ in range(reps):
                lg, state["s"] = model.forward_step(state["cur"], state["s"])
                state["cur"] = lg[0].argmax().view(1)

    run()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        ts.append((time.perf_counter() - t0) / reps * 1e3)
    return {"kernel": kernel, "threads": threads, "T": T,
            "ms_per_step": sorted(ts)[len(ts) // 2]}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        k, th, T, reps = map(int, sys.argv[2:6])
        print(json.dumps(child(k, th, T, reps)))
        return 0
    env = {**os.environ, "SUNNY_W64": "F:/devtools/w64devkit"}
    rows = []
    for th in (1, 8):
        for T in (256, 1024):
            for k in (0, 1):
                r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                    "--child", str(k), str(th), str(T), "30"],
                                   capture_output=True, text=True, cwd=str(ROOT), env=env)
                try:
                    rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
                except Exception:
                    print(f"  失败 k={k} th={th} T={T}: {(r.stderr or r.stdout or '')[-300:]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'线程':>4} {'T':>6} {'torch ms':>10} {'内核 ms':>10} {'比值':>8}")
    print("-" * 42)
    for th in (1, 8):
        for T in (256, 1024):
            a = next((d for d in rows if d["threads"] == th and d["T"] == T and d["kernel"] == 0), None)
            b = next((d for d in rows if d["threads"] == th and d["T"] == T and d["kernel"] == 1), None)
            if a and b:
                print(f"{th:>4} {T:>6} {a['ms_per_step']:>10.4f} {b['ms_per_step']:>10.4f} "
                      f"{a['ms_per_step']/b['ms_per_step']:>7.3f}x")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())