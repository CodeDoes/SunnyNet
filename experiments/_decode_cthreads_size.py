"""C 解码后端的线程数应由模型规模决定, 而非一律用物理核数.

小模型权重可驻留缓存, OpenMP 并行无带宽收益、只有同步开销; 大模型访存受限才需要并行。
本脚本扫 4 个规模档 x 线程数, 给出拐点。
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
SIZES = {"d80": (80, 160, 244, 4), "d128": (128, 192, 64, 4),
         "d192": (192, 320, 96, 4), "d256": (256, 448, 128, 4),
         "d384": (384, 640, 192, 4), "d512": (512, 896, 320, 6),
         "d768": (768, 1280, 448, 6)}
THREADS = [1, 2, 4, 6]
REPS = 40

CELL = r'''
import json, sys, time
sys.path.insert(0, r"{root}")
import torch, statistics as st
torch.set_num_threads(1); torch.manual_seed(1234)
from sunnynet.runtime.model import SnModel, set_threads
m = SnModel(r"{blob}", {V}, {d}, {i}, {r}, 4, {L}, 8, 1)
set_threads({thr})
tok = [1234, 5678, 910]
m.prefill(tok)
one = 777; m.step(one)
ts = []
for _ in range({reps}):
    t0 = time.perf_counter(); m.step(one); ts.append(time.perf_counter()-t0)
print("CELL " + json.dumps({{"ms": st.median(ts) * 1000, "mb": {mb:.1f}}}))
'''


def build_and_export(tag):
    import torch
    from sunnynet.model import SunnyConfig, SunnyNet
    d, i, r, L = SIZES[tag]
    ck = ROOT / f"experiments/ckpts_decode/{tag}.pt"
    blob = ROOT / f"experiments/ckpts_decode/{tag}.runtime.bin"
    if not ck.exists():
        torch.manual_seed(1234)
        c = SunnyConfig(vocab_size=V, dim=d, inner=i, rank=r, layers=L, h_max=2048.0)
        c.scan_chunk = 64
        m = SunnyNet(c)
        if any(b._need_clamp for b in m.blocks):
            raise SystemExit(f"{tag} 需要 clamp, C 后端不支持")
        torch.save({"arch": "sunnynet", "model": m.state_dict(),
                    "config": c.__dict__ if hasattr(c, "__dict__") else c}, ck)
    if not blob.exists():
        from experiments._export_runtime import export
        export(str(ck), str(blob))
    return ck, blob


def run_child(blob, spec, mb, thr):
    d, i, r, L = spec
    code = CELL.format(root=str(ROOT), blob=blob, V=V, d=d, i=i, r=r, L=L,
                       thr=thr, reps=REPS, mb=mb)
    rr = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                        text=True, encoding="utf-8", cwd=str(ROOT), env=dict(os.environ))
    for ln in rr.stdout.splitlines():
        if ln.startswith("CELL "):
            return json.loads(ln[5:])
    raise RuntimeError((rr.stderr or rr.stdout or "")[-900:])


def main():
    sys.path.insert(0, str(ROOT))
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    res = {}
    print(f"{'model':8s}{'MB':>8s}" + "".join(f"thr{t}".ljust(11) for t in THREADS) + "  best")
    for tag, spec in SIZES.items():
        ck, blob = build_and_export(tag)
        mb = blob.stat().st_size / 1e6
        acc = {t: [] for t in THREADS}
        for rd in range(rounds):
            for t in THREADS:
                acc[t].append(run_child(blob, spec, mb, t)["ms"])
        row = {t: st.median(acc[t]) for t in THREADS}
        res[tag] = {"MB": mb, "ms": row,
                    "best": min(row, key=row.get)}
        best = min(row, key=row.get)
        print(f"{tag:8s}{mb:8.1f}" + "".join(f"{row[t]:.3f}".ljust(11) for t in THREADS)
              + f"  thr{best} ({row[1]/row[best]:.2f}x vs 1)")
    out = ROOT / "experiments/results/_decode_cthreads_size.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())