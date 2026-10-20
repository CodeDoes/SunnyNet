"""顺序扫描 (dim, layers, inner) 网格, 每格独立子进程.

用法: python -X utf8 experiments/_cfg_drive.py [T] [B] [reps]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
T = sys.argv[1] if len(sys.argv) > 1 else "256"
B = sys.argv[2] if len(sys.argv) > 2 else "8"
REPS = sys.argv[3] if len(sys.argv) > 3 else "20"

TARGET = 365844
GRID = [
    (128, 4, 192),
    (112, 4, 192),
    (104, 4, 192),
    (96, 4, 192),
    (96, 4, 160),
    (96, 5, 176),
    (88, 4, 176),
    (88, 4, 192),
    (88, 5, 176),
    (80, 4, 192),
    (80, 4, 160),
    (80, 5, 160),
    (72, 5, 160),
    (72, 6, 144),
    (64, 6, 160),
]

rows = []
for dim, L, inner in GRID:
    r = subprocess.run([PY, "-X", "utf8", "experiments/_cfg_scan.py",
                        str(dim), str(L), str(TARGET), str(inner), T, B, REPS],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        print(f"  dim={dim} L={L} inner={inner} FAILED: {r.stderr[-200:]}")
        continue
    d = json.loads(line[0][7:])
    rows.append(d)
    print(f"  dim={dim:3d} L={L} inner={inner:3d} rank={d['rank']:3d} "
          f"stack={d['stack']:,} ({d['bias_pct']:+.2f}%) "
          f"{d['train_ms']:7.1f}ms {d['ms_per_token']:.5f} ms/tok")

rows.sort(key=lambda r: r["ms_per_token"])
print(f"\n=== T={T} 按 ms/tok 排序 ===")
for r in rows:
    print(f"  dim={r['dim']:3d} L={r['layers']} inner={r['inner']:3d} rank={r['rank']:3d} "
          f"stack={r['stack']:,} {r['ms_per_token']:.5f} ms/tok")
Path("experiments/results/_cfg_drive.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")