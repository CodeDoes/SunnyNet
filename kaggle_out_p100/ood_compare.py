"""Held-out OOD PPL: 94M model vs 21M model vs unigram baseline.

All measured on the same two unseen books, each with its OWN tokenizer.
The unigram baseline (next-word-from-histogram, no model) is the floor: if a
small model beats it by little, the model has learned almost nothing beyond
word frequency."""

import math
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.tokenizer import BPETokenizer

HOLD = [
    ROOT / "data/proc/heldout_4324.txt",
    ROOT / "data/proc/heldout_5147.txt",
]
_HOLD_IDS = (4324, 5147)


def ensure_heldout() -> None:
    """Held-out books live under gitignored data/; fetch them if absent."""
    ROOT.joinpath("data", "proc").mkdir(parents=True, exist_ok=True)
    for gid in _HOLD_IDS:
        p = ROOT / f"data/proc/heldout_{gid}.txt"
        if not p.exists() or p.stat().st_size < 1000:
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            try:
                p.write_bytes(urlopen(url, timeout=60).read())
                print(f"  fetched held-out book {gid}")
            except URLError as e:  # keep any cached copy, else fail loudly
                if not p.exists():
                    raise SystemExit(f"could not fetch held-out book {gid}: {e}")
                print(f"  [warn] using cached held-out book {gid} (fetch failed: {e})")


def text() -> str:
    ensure_heldout()
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in HOLD)


def load(ckpt, tok_path):
    s = torch.load(str(ROOT / ckpt), map_location="cpu", weights_only=False)
    m = SunnyNet(SunnyConfig(**s["config"]))
    m.load_state_dict(s["model"])
    m.eval().cpu()
    return m, BPETokenizer.load(str(ROOT / tok_path))


def model_metrics(m, tk, nwin=40, win=256, seed=0):
    ids = tk.encode(text())
    rng = random.Random(seed)
    tot_ce, tot_n, hits = 0.0, 0, 0
    for _ in range(nwin):
        off = rng.randrange(win, len(ids) - win)
        chunk = ids[off : off + win + 1]
        x = torch.tensor([chunk[:-1]])
        with torch.no_grad():
            logits = m(x)[0]  # (T, V)
        tgt = torch.tensor(chunk[1:])
        ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        tot_ce += ce.sum().item()
        tot_n += ce.numel()
        hits += int((logits.argmax(-1) == tgt).sum().item())
    return math.exp(tot_ce / tot_n), hits / tot_n


def unigram_baseline(tk):
    """Entropy of the token histogram of the held-out text (bits/token)."""
    ids = tk.encode(text())
    c = Counter(ids[1:])
    tot = sum(c.values())
    bits = -sum(cnt / tot * math.log2(cnt / tot) for cnt in c.values())
    return 2**bits


def main():
    print("Held-out text (2 unseen books, 338K chars). OOD PPL + top-1:\n")
    m94, tk94 = load("runs/p100m_en.pt", "runs/p100m_en_tokenizer.json")
    p94, a94 = model_metrics(m94, tk94)
    print(
        f"94M p100m (P100, ~100M tok, 35 books):  PPL={p94:8.1f}  top1={a94 * 100:5.1f}%"
    )

    m21, tk21 = load("runs/proto20m_en.pt", "data/proc/tokenizer.json")
    p21, a21 = model_metrics(m21, tk21)
    print(
        f"21M proto20m (GPU, ~20M tok):           PPL={p21:8.1f}  top1={a21 * 100:5.1f}%"
    )

    ub94 = unigram_baseline(tk94)
    ub21 = unigram_baseline(tk21)
    print(
        f"unigram (no model), 94M tokenizer:      PPL={ub94:8.1f}  (top1 = best-word freq)"
    )
    print(f"unigram (no model), 21M tokenizer:     PPL={ub21:8.1f}")


if __name__ == "__main__":
    main()
