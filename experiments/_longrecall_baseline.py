"""当前架构的长程回召基线, 不重叠窗口覆盖整个验证集.

§8 的 ckpt 出自 §12 修正前的旧架构 (state_dict 无 theta/delta_max/delta_min),
且原探针用 6 个随机窗口, en_literature far 桶只有 433 样本, 95% CI 半宽 0.0063,
检不出 <0.05 的效应. 本脚本改为按 seq_len 不重叠铺满验证集, 把英语 far 桶样本量
拉到验证集上限, 让 A/B 有足够功效.

用法:
  python -X utf8 experiments/_longrecall_baseline.py <ckpt> [seq_len]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_corpus import DOMAINS, load_tokens  # noqa: E402
from train.evaluate import BUCKETS, load_model  # noqa: E402


def _windows(a: np.ndarray, seq_len: int):
    """不重叠窗口, 覆盖到验证集尾. 相邻窗口起点相距 seq_len."""
    n = a.size // seq_len
    for i in range(n):
        yield a[i * seq_len:(i + 1) * seq_len].astype(np.int64)


@torch.no_grad()
def recall_nonoverlap(model, domain: str, seq_len: int, vocab: int,
                      ngram: int = 1, max_distance: int = 3072) -> dict:
    """与 train.evaluate.long_range_recall 同口径, 但窗口不重叠."""
    a = load_tokens(domain, "val")
    stats = {name: [0, 0] for _, _, name in BUCKETS}
    model.eval()
    n = max(1, ngram)
    n_win = 0
    for seq in _windows(a, seq_len):
        if seq.size < n + 2:
            continue
        n_win += 1
        last_seen: dict[tuple, int] = {}
        logits = model(torch.from_numpy(seq)[None, :])[0]
        for t in range(n, seq.size):
            key = tuple(int(v) for v in seq[t - n:t])
            prev = last_seen.get(key)
            if prev is not None:
                dist = t - prev
                if dist <= max_distance:
                    target = int(seq[t])
                    pred = int(logits[t].argmax())
                    for lo, hi, name in BUCKETS:
                        if lo <= dist <= hi:
                            stats[name][0] += 1
                            stats[name][1] += int(pred == target)
                            break
            last_seen[key] = t
    out = {"n_windows": n_win}
    for name in stats:
        hit, tot = stats[name][1], stats[name][0]
        out[name] = {"hit": hit, "total": tot,
                     "acc": round(hit / tot, 4) if tot else None}
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    ckpt = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    model, cfg = load_model(ckpt)
    vocab = cfg.vocab_size
    tag = Path(ckpt).stem
    res = {"ckpt": ckpt, "seq_len": T, "ngram": {}}
    for n in (1, 4, 8):
        res["ngram"][f"ngram{n}"] = {}
        for d in DOMAINS:
            r = recall_nonoverlap(model, d, T, vocab, ngram=n)
            for b in ("near", "mid", "far"):
                if r[b]["total"]:
                    r[b]["ci95"] = wilson(r[b]["hit"], r[b]["total"])
            res["ngram"][f"ngram{n}"][d] = r
            print(f"  {tag} ngram{n} {d:14s} win={r['n_windows']:3d} "
                  f"near={r['near']['acc']} mid={r['mid']['acc']} far={r['far']['acc']}"
                  f" far_n={r['far']['total']} far_ci={r['far'].get('ci95')}")
    out = ROOT / "experiments" / "results" / f"_longrecall_base_{tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()