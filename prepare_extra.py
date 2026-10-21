"""独立新域语料采集: 与原训练三域 (zh_ancient/en_literature/code_python) 无重叠.

用于跨分布泛化测量. 域:
  arxiv_en      英文科学摘要 (arxiv API, cs.CL/LG/AI/stat.ML)
  gutenberg_fr  法文文学 (Project Gutenberg 法语书目, 与原集不同书)
  zh_tech_news  简体中文科技新闻 (新华网 tech/edu/health 正文)

产出: data/raw_ext/<domain>/*.txt
不重训 tokenizer: 复用 data/proc/tokenizer.json, 简繁未登录率另行统计.
"""
from __future__ import annotations

import argparse
import re
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "data" / "raw_ext"
UA = {"User-Agent": "Mozilla/5.0 (sunnynet-corpus-fetch)"}
_NV = ssl.create_default_context()
_NV.check_hostname = False
_NV.verify_mode = ssl.CERT_NONE

ARXIV_CATS = ["cs.CL", "cs.LG", "cs.AI", "stat.ML"]
GUT_FR_BOOKS = [17489, 17989, 18217, 22432, 26635, 30807, 31714, 40042]
NEWS_CHANNELS = {
    "xinhua_tech": "https://www.news.cn/tech/",
    "xinhua_edu": "https://education.news.cn/",
    "xinhua_health": "https://www.news.cn/health/",
    "xinhua_culture": "https://www.news.cn/culture/",
    "gmw_news": "https://news.gmw.cn/",
    "gmw_home": "https://www.gmw.cn/",
    "people_edu": "http://edu.people.com.cn/",
    "people_scitech": "http://scitech.people.com.cn/",
    "cas": "https://www.cas.cn/",
}

# 政治类正文不进入语料 (内容准则), 采集阶段直接丢弃
POLITICAL = re.compile(
    r"习近平|总书记|中共中央|国务院总理|政治局|全国人大|全国政协|"
    r"外交部|国防部|中央军委|党和国家|党的领导|社会主义|马克思|"
    r"党委|党建|巡视|纪检|意识形态")


def fetch(url: str, timeout: int = 25, noverify: bool = False) -> bytes | None:
    """IncompleteRead 的 partial 也接受: Gutenberg 大文件常被中途断开.

    不要重试 3 次: 大文件每次都要重传, 累计耗时会把进程拖到被杀.
    """
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_NV if noverify else None) as r:
            return r.read()
    except urllib.error.HTTPError:
        return None
    except Exception as e:
        return getattr(e, "partial", None)


def fetch_stream(url: str, timeout: int = 90, noverify: bool = False) -> bytes | None:
    """Gutenberg 大文件按块读: 中途断开时保留已读部分, 不重传整个文件."""
    buf = bytearray()
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_NV if noverify else None) as r:
            while True:
                try:
                    chunk = r.read(65536)
                except Exception:
                    break
                if not chunk:
                    break
                buf.extend(chunk)
    except Exception:
        return bytes(buf) if buf else None
    return bytes(buf)


def decode(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", "ignore")


def gather_arxiv(pages: int, per_page: int) -> list[str]:
    docs = []
    for cat in ARXIV_CATS:
        for p in range(pages):
            url = (f"http://export.arxiv.org/api/query?search_query=cat:{cat}"
                   f"&start={p * per_page}&max_results={per_page}")
            raw = fetch(url)
            if not raw:
                continue
            for m in re.finditer(r"(?s)<entry>(.*?)</entry>", decode(raw)):
                e = m.group(1)
                su = re.search(r"(?s)<summary>(.*?)</summary>", e)
                if not su:
                    continue
                ti = re.search(r"(?s)<title>(.*?)</title>", e)
                s = re.sub(r"\s+", " ", su.group(1)).strip()
                t = re.sub(r"\s+", " ", ti.group(1)).strip() if ti else ""
                docs.append(f"{t}\n{s}\n")
            print(f"  arxiv {cat} page{p} -> 累计 {len(docs)} 篇", flush=True)
    return docs


def strip_gutenberg(text: str) -> str:
    m = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text)
    if m:
        text = text[:m.start()]
    return text


def gather_gutenberg_fr(books: list[int]) -> list[str]:
    docs = []
    for bid in books:
        raw = fetch_stream(f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                           noverify=True)
        if not raw:
            print(f"  gut_fr {bid} 失败", flush=True)
            continue
        t = strip_gutenberg(decode(raw))
        if len(t) < 5000:
            continue
        docs.append(t)
        print(f"  gut_fr {bid} -> {len(t)} 字符", flush=True)
    return docs


def article_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    m = re.search(r'(?is)<div[^>]*(?:id|class)="[^"]*(?:detail|content|article|main|text)[^"]*"[^>]*>(.*)',
                  html)
    seg = m.group(1) if m else html
    ps = re.findall(r"(?is)<p[^>]*>(.*?)</p>", seg)
    t = " ".join(re.sub(r"(?s)<[^>]+>", "", p) for p in ps)
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def gather_news(max_articles: int) -> list[str]:
    links: list[str] = []
    for name, home in NEWS_CHANNELS.items():
        raw = fetch(home)
        if not raw:
            print(f"  news {name} 首页失败", flush=True)
            continue
        h = decode(raw)
        found = [x for x in dict.fromkeys(re.findall(r'href="(https?://[^"]+)"', h))
                 if re.search(r"20\d\d", x) and x.endswith((".html", ".htm", ".shtml"))
                 and "index" not in x]
        links.extend(found[:max_articles])
        print(f"  news {name} -> {len(found)} 链接", flush=True)
    links = list(dict.fromkeys(links))
    print(f"  合计候选 {len(links)}", flush=True)

    docs: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, u): u for u in links}
        for f in as_completed(futs):
            raw = f.result()
            if not raw:
                continue
            t = article_text(decode(raw))
            if len(t) >= 400 and not POLITICAL.search(t):
                docs.append(t)
    print(f"  新闻正文 -> {len(docs)} 篇 (已剔除政治类)", flush=True)
    return docs


def write_domain(name: str, docs: list[str], per_file: int = 200_000) -> int:
    d = EXT / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.txt"):
        f.unlink()
    total, idx, buf = 0, 0, []
    for text in docs:
        buf.append(text)
        if sum(len(x) for x in buf) >= per_file:
            (d / f"{idx:04d}.txt").write_text("\n\n".join(buf), encoding="utf-8")
            total += sum(len(x) for x in buf)
            idx += 1
            buf = []
    if buf:
        (d / f"{idx:04d}.txt").write_text("\n\n".join(buf), encoding="utf-8")
        total += sum(len(x) for x in buf)
    print(f"[{name}] {total:,} 字符 -> {len(list(d.glob('*.txt')))} 文件")
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="arxiv_en,gutenberg_fr,zh_tech_news")
    ap.add_argument("--arxiv-pages", type=int, default=4)
    ap.add_argument("--news-articles", type=int, default=220)
    args = ap.parse_args()
    want = set(args.domains.split(","))

    if "arxiv_en" in want:
        write_domain("arxiv_en", gather_arxiv(args.arxiv_pages, 100))
    if "gutenberg_fr" in want:
        write_domain("gutenberg_fr", gather_gutenberg_fr(GUT_FR_BOOKS))
    if "zh_tech_news" in want:
        write_domain("zh_tech_news", gather_news(args.news_articles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())