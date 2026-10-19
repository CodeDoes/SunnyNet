"""真实语料采集, 不生成任何合成文本.

来源: Project Gutenberg 中文书目页真实解析 (不硬编码 ID) + TheAlgorithms/Python tarball.
分域规则: 汉字占比 >= 0.20 归 zh_ancient, 拉丁字母占比 >= 0.60 归 en_literature, 否则丢弃.
中文维基在本机网络不可达, 故"百科"域缺失.

产出: data/raw/{zh_ancient,en_literature,code_python}/*.txt
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
UA = {"User-Agent": "Mozilla/5.0 (sunnynet-corpus-fetch)"}
CATALOG_URL = "https://www.gutenberg.org/browse/languages/zh"
CATALOG_PATH = CACHE / "gutenberg_zh_catalog.html"

EXTRA_EN_BOOKS = [1342, 2701, 84, 1661, 98, 2600, 76, 345, 132, 158, 11, 174]
CODE_REPOS = [("TheAlgorithms/Python", "master")]

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN = re.compile(r"[A-Za-z]")


def fetch(url: str, timeout: int = 45) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def decode_any(data: bytes) -> str | None:
    for enc in ("utf-8", "gb18030", "big5", "utf-16", "latin-1"):
        try:
            t = data.decode(enc)
            if t.count("\ufffd") < len(t) * 0.02:
                return t
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def load_catalog_ids() -> list[int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    if not CATALOG_PATH.exists():
        data = fetch(CATALOG_URL, timeout=60)
        if not data:
            raise SystemExit("无法获取 Gutenberg 中文书目页")
        CATALOG_PATH.write_bytes(data)
    html = CATALOG_PATH.read_text(encoding="utf-8", errors="ignore")
    return sorted({int(x) for x in re.findall(r"ebooks/(\d+)", html)})


def gutenberg_text(book_id: int) -> str | None:
    for url in (f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"):
        data = fetch(url)
        if data and len(data) > 2000:
            t = decode_any(data)
            if t:
                return t
    return None


def strip_boilerplate(text: str) -> str:
    m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text, re.S)
    if m:
        text = text[:m.start()]
    return text.strip()


def classify(text: str) -> str:
    body = re.sub(r"\s+", "", text)
    if len(body) < 4000:
        return "drop"
    han = len(HAN.findall(body)) / len(body)
    latin = len(LATIN.findall(body)) / len(body)
    if han >= 0.20:
        return "zh_ancient"
    if latin >= 0.60:
        return "en_literature"
    return "drop"


def clean_zh(text: str) -> str:
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t\u3000]+", "", ln.strip())
        if len(ln) < 4:
            continue
        han = len(HAN.findall(ln))
        if han == 0 or han / len(ln) < 0.35:
            continue
        out.append(ln)
    return "\n".join(out)


def clean_en(text: str) -> str:
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln.strip())
        if len(ln) < 25:
            continue
        if len(LATIN.findall(ln)) / len(ln) < 0.55:
            continue
        out.append(ln)
    return "\n".join(out)


def clean_code(text: str) -> str:
    text = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", text)
    lines = [ln.rstrip() for ln in text.splitlines() if not ln.strip().startswith("#")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def save(domain: str, name: str, text: str) -> int:
    d = RAW / domain
    d.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".txt"):
        name += ".txt"
    (d / name).write_text(text, encoding="utf-8")
    return len(text)


def collect_books(ids: list[int], workers: int = 8):
    stats = {"zh_ancient": 0, "en_literature": 0, "drop": 0}
    chars = {"zh_ancient": 0, "en_literature": 0}
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(gutenberg_text, i): i for i in ids}
        for f in as_completed(futs):
            bid = futs[f]
            done += 1
            raw = f.result()
            if not raw:
                stats["drop"] += 1
                continue
            body = strip_boilerplate(raw)
            dom = classify(body)
            if dom == "drop":
                stats["drop"] += 1
                continue
            body = clean_zh(body) if dom == "zh_ancient" else clean_en(body)
            if len(body) < 8000:
                stats["drop"] += 1
                continue
            chars[dom] += save(dom, f"gutenberg_{bid}.txt", body)
            stats[dom] += 1
            if done % 60 == 0:
                print(f"  ...{done}/{len(ids)}  zh={stats['zh_ancient']} en={stats['en_literature']}")
    return stats, chars


def collect_code(taps: int = 4):
    total, n = 0, 0
    for repo, branch in CODE_REPOS:
        data = fetch(f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}", timeout=120)
        if not data:
            print(f"  [skip] {repo}")
            continue
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(".py") or m.size > 100_000:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                src = decode_any(fh.read())
                if not src:
                    continue
                body = clean_code(src)
                if len(body) < 200:
                    continue
                total += save("code_python", m.name.replace("/", "__"), body)
                n += 1
        print(f"  {repo}: {n} files")
    return n, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-books", type=int, default=0, help="0 = 全部")
    ap.add_argument("--skip-code", action="store_true")
    ap.add_argument("--only-code", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if not args.only_code:
        ids = load_catalog_ids()
        if args.max_books:
            ids = ids[:args.max_books]
        ids = ids + EXTRA_EN_BOOKS
        print(f"[1/2] Gutenberg 书籍 {len(ids)} 本 (目录页真实解析 + {len(EXTRA_EN_BOOKS)} 本经典英文)")
        stats, chars = collect_books(ids, args.workers)
        print(f"  分域结果: zh_ancient={stats['zh_ancient']} en_literature={stats['en_literature']} drop={stats['drop']}")
        print(f"  字符数:   zh={chars['zh_ancient']:,} en={chars['en_literature']:,}")

    if not args.skip_code:
        print("[2/2] Python 源码")
        n, c = collect_code()
        print(f"  code_python: files={n} chars={c:,}")

    print("\n语料统计")
    for d in sorted(RAW.iterdir()):
        if d.is_dir():
            fs = list(d.glob("*.txt"))
            print(f"  {d.name:16s} files={len(fs):5d} bytes={sum(f.stat().st_size for f in fs):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())