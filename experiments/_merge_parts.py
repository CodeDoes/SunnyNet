"""分块下载全部完成后合并为单文件."""
from __future__ import annotations

import sys
from pathlib import Path

parts_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
files = sorted(parts_dir.glob("*.part"))
print(f"合并 {len(files)} 块 -> {out}", flush=True)
with open(out, "wb") as o:
    for f in files:
        o.write(f.read_bytes())
print(f"OK {out.stat().st_size} bytes ({out.stat().st_size/2**20:.1f} MB)", flush=True)