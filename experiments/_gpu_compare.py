"""汇总 CPU vs GPU 对照 (调用 _gpu_probe 的子进程, 每格独立)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_PY = os.environ.get("CPU_PYTHON_BIN", "python")
GPU_PY = os.environ.get("GPU_PYTHON_BIN", "python")

CASES = [("small", 512), ("small", 1024), ("proto20mw", 512), ("proto20mw", 1024)]
out = []


def run(py: str, dev: str, preset: str, T: int) -> dict:
    p = subprocess.run(
        [py, "-X", "utf8", str(ROOT / "experiments" / "_gpu_probe.py"), dev, preset, str(T), "8", "3"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1800,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": p.stderr[-200:]}


for preset, T in CASES:
    c = run(CPU_PY, "cpu", preset, T)
    g = run(GPU_PY, "cuda", preset, T)
    row = {"preset": preset, "T": T, "cpu": c, "gpu": g}
    if "fwd_ms_median" in c and "fwd_ms_median" in g:
        row["fwd_speedup"] = round(c["fwd_ms_median"] / g["fwd_ms_median"], 2)
        row["step_speedup"] = round(c["step_ms_median"] / g["step_ms_median"], 2)
    out.append(row)
    print(f"{preset} T={T}: fwd {row.get('fwd_speedup')}x step {row.get('step_speedup')}x", flush=True)

p = ROOT / "experiments" / "results" / "gpu_compare.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", p)