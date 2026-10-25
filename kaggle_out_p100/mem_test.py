"""In-domain vs held-out PPL + top-1 next-token accuracy for the 94M p100m model.

Uses the model's OWN tokenizer (runs/p100m_en_tokenizer.json).
  IN-DOMAIN  proxy: the local classics the model was trained on (data/raw/en_literature)
  HELD-OUT:         freshly-downloaded books it has never seen (gutenberg 4324/5147)

A large PPL gap (in-domain << held-out) means the good-looking loss is
memorization, not language. Top-1 tells us the true next-word accuracy."""

import math
import random
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sunnynet.model import SunnyConfig, SunnyNet
from sunnynet.tokenizer import BPETokenizer

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


def load(ckpt, tok_path):
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = SunnyNet(SunnyConfig(**s["config"]))
    m.load_state_dict(s["model"])
    m.eval().cpu()
    return m, BPETokenizer.load(tok_path)


def metrics(model, tk, text, nwin=40, win=256, seed=0):
    ids = tk.encode(text)
    if len(ids) < win + 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    tot_ce, tot_n, hits = 0.0, 0, 0
    for _ in range(nwin):
        off = rng.randrange(win, len(ids) - win)
        chunk = ids[off : off + win + 1]
        x = torch.tensor([chunk[:-1]])
        with torch.no_grad():
            logits = model(x)[0]  # (T, V)
        tgt = torch.tensor(chunk[1:])  # (T,)
        ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        tot_ce += ce.sum().item()
        tot_n += ce.numel()
        hits += int((logits.argmax(-1) == tgt).sum().item())
    return math.exp(tot_ce / tot_n), hits / tot_n


def main():
    m, tk = load(
        str(ROOT / "runs/p100m_en.pt"), str(ROOT / "runs/p100m_en_tokenizer.json")
    )

    # IN-DOMAIN proxy: model was trained on ~35 of these classics (gitignored)
    in_dir = ROOT / "data/raw/en_literature"
    in_dom = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(in_dir.glob("*.txt"))
    )
    have_in = bool(in_dom.strip())
    if not have_in:
        print(f"  [warn] in-domain dir {in_dir} empty/absent; skipping in-domain PPL")

    # HELD-OUT: freshly downloaded, never seen
    ensure_heldout()
    held = (
        (ROOT / "data/proc/heldout_4324.txt").read_text(errors="replace")
        + "\n"
        + (ROOT / "data/proc/heldout_5147.txt").read_text(errors="replace")
    )

    if have_in:
        ip, ia = metrics(m, tk, in_dom)
        print(
            f"IN-DOMAIN  (trained-on classics): PPL={ip:.2f}   top1_next_word_acc={ia * 100:.1f}%"
        )
    hp, ha = metrics(m, tk, held)
    print(
        f"HELD-OUT   (unseen books):         PPL={hp:.2f}   top1_next_word_acc={ha * 100:.1f}%"
    )
    if have_in:
        print(
            f"PPL gap held/in = {hp / ip:.2f}x   |  top-1 drop = {(ia - ha) * 100:.1f} pts"
        )


if __name__ == "__main__":
    main()
