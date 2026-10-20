"""为单个分块续传缺失字节 (反复重试直到补满)."""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

URL = sys.argv[1]
PART = Path(sys.argv[2])
START = int(sys.argv[3])
END = int(sys.argv[4])

opener = urllib.request.build_opener(urllib.request.ProxyHandler())
need = END - START + 1
print(f"目标: bytes {START}-{END} ({need/2**20:.1f} MB)", flush=True)

for attempt in range(400):
    have = PART.stat().st_size if PART.exists() else 0
    if have >= need:
        print(f"完成 {have} bytes", flush=True)
        break
    s = START + have
    try:
        req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{END}"})
        with opener.open(req, timeout=45) as r:
            mode = "ab" if have else "wb"
            with open(PART, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        print(f"  [{attempt}] {str(e)[:70]}", flush=True)
        time.sleep(2)
    if attempt % 20 == 0:
        have = PART.stat().st_size if PART.exists() else 0
        print(f"  [{attempt}] {have/2**20:.1f}/{need/2**20:.1f} MB", flush=True)

have = PART.stat().st_size if PART.exists() else 0
print(f"最终 {have}/{need} bytes", flush=True)