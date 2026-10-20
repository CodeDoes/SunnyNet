"""多线程分块下载 (Range 并发), 用于从慢速源拉大 wheel.

用法: python _fastdl.py <url> <out> [threads]
"""
from __future__ import annotations

import os
import sys
import threading
import urllib.request
from pathlib import Path

URL = sys.argv[1]
OUT = Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

opener = urllib.request.build_opener(urllib.request.ProxyHandler())


def head_size() -> int:
    req = urllib.request.Request(URL, method="HEAD")
    with opener.open(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


total = head_size()
OUT.parent.mkdir(parents=True, exist_ok=True)
part_dir = OUT.parent / (OUT.name + ".parts")
part_dir.mkdir(exist_ok=True)
print(f"total={total} ({total/2**20:.1f} MB), threads={N}", flush=True)

bounds = []
step = total // N
for i in range(N):
    a = i * step
    b = total - 1 if i == N - 1 else (i + 1) * step - 1
    bounds.append((a, b))

lock = threading.Lock()
done = [0]


def worker(i: int, a: int, b: int):
    fp = part_dir / f"{i:02d}.part"
    have = fp.stat().st_size if fp.exists() else 0
    start = a + have
    if start > b:
        with lock:
            done[0] += b - a + 1
        return
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{b}"})
    mode = "ab" if have else "wb"
    try:
        with opener.open(req, timeout=60) as r, open(fp, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                with lock:
                    done[0] += len(chunk)
    except Exception as e:
        print(f"  [{i}] 错误(将重试): {e}", flush=True)


for attempt in range(6):
    if all((part_dir / f"{i:02d}.part").exists()
           and (part_dir / f"{i:02d}.part").stat().st_size == (bounds[i][1] - bounds[i][0] + 1)
           for i in range(N)):
        break
    threads = [threading.Thread(target=worker, args=(i, a, b)) for i, (a, b) in enumerate(bounds)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = sum((part_dir / f"{i:02d}.part").stat().st_size
              for i in range(N) if (part_dir / f"{i:02d}.part").exists())
    print(f"  attempt {attempt}: {got/2**20:.1f}/{total/2**20:.1f} MB", flush=True)
    if got >= total:
        break

with open(OUT, "wb") as out:
    for i in range(N):
        fp = part_dir / f"{i:02d}.part"
        out.write(fp.read_bytes())
print(f"OK {OUT} {OUT.stat().st_size} bytes", flush=True)