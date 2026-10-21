"""C 全模型解码 vs torch forward_step 的端到端延迟对照 (每格独立子进程).

口径: B=1 单序列, 单线程, 先走满 prefill + 预热再测单步, 3 轮取中位.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "experiments/results/_runtime_decode.json"
ROUNDS = 3


def child(kind: str, prefill: int, steps: int) -> dict:
    import time
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(1)
    from sunnynet.cli import load_model
    m, cfg, _ = load_model(str(ROOT / "experiments/ckpts_ood/sunnynet.pt"))
    m.eval()
    torch.manual_seed(11)
    toks = torch.randint(0, cfg.vocab_size, (prefill + steps * 2,)).tolist()

    if kind == "c":
        from sunnynet.runtime.model import SnModel
        rt = SnModel(ROOT / "experiments/ckpts_ood/sunnynet.runtime.bin",
                     cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank, cfg.taps,
                     cfg.layers, 8, int(cfg.tie_embeddings))
        rt.prefill(toks[:prefill])
        for t in toks[prefill:prefill + steps]:
            rt.step(t)
        t0 = time.perf_counter()
        for t in toks[prefill + steps:prefill + 2 * steps]:
            rt.step(t)
        dt = time.perf_counter() - t0
    else:
        s = m.init_state(1)
        with torch.no_grad():
            lg, s = m(torch.tensor([toks[:prefill]]), return_state=True)
            for t in toks[prefill:prefill + steps]:
                lg, s = m.forward_step(torch.tensor([t]), s)
            t0 = time.perf_counter()
            for t in toks[prefill + steps:prefill + 2 * steps]:
                lg, s = m.forward_step(torch.tensor([t]), s)
            dt = time.perf_counter() - t0
    return {"kind": kind, "prefill": prefill, "steps": steps,
            "ms_per_step": dt / steps * 1e3}


def main():
    cfgs = [(256, 30), (1024, 30), (2048, 30)]
    raw = {}
    for rnd in range(ROUNDS):
        for prefill, steps in cfgs:
            for kind in ("torch", "c"):
                r = subprocess.run([PY, "-X", "utf8", str(Path(__file__).resolve()),
                                    "--child", kind, str(prefill), str(steps)],
                                   capture_output=True, text=True, cwd=str(ROOT),
                                   env={**os.environ, "SUNNY_W64": os.environ.get(
                                       "SUNNY_W64", "F:/devtools/w64devkit")})
                try:
                    d = json.loads(r.stdout.strip().splitlines()[-1])
                    raw.setdefault((prefill, kind), []).append(d["ms_per_step"])
                except Exception:
                    print(f"  失败 r{rnd} {kind} prefill={prefill}: "
                          f"{(r.stderr or r.stdout or '')[-200:]}")
    rows = []
    print(f"{'prefill':>8} {'torch 中位':>11} {'C 中位':>9} {'比值':>7} {'逐轮占优':>9}")
    print("-" * 50)
    for prefill, _ in cfgs:
        a = sorted(raw[(prefill, "torch")])
        b = sorted(raw[(prefill, "c")])
        if not a or not b:
            continue
        ma, mb = a[len(a) // 2], b[len(b) // 2]
        win = sum(1 for x, y in zip(a, b) if y < x)
        n = min(len(a), len(b))
        print(f"{prefill:>8} {ma:>11.3f} {mb:>9.3f} {ma/mb:>6.2f}x "
              f"{win}/{n}")
        rows.append({"prefill": prefill, "torch_median": ma, "c_median": mb,
                     "ratio": ma / mb, "c_wins": win, "rounds": n,
                     "torch_all": a, "c_all": b})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
        raise SystemExit(0)
    raise SystemExit(main())