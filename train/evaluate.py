"""评估: 困惑度 / 长距离依赖 / 生成样例.

长距离依赖探针 (LongRangeRecall) 的设计:
  在真实验证语料上找"重复 n-gram"上下文: 若当前位置 t 的前缀 token a 曾经与
  某个后继 b 一起出现过 (即 (a,b) 在 t 之前出现过), 则模型若真能利用长程上下文,
  就应把 b 排在 top-1. 按"最近一次出现距离"分桶统计命中率:

      near  : 距离 <= 16
      mid   : 17 ~ 256
      far   : 257 ~ 2048
      ultra : > 2048

  由于完全取自真实语料, 不含任何人造构造, 分桶命中率可直接横向比较两个架构
  在长程信息保持上的差异. 距离越远命中率越高, 说明状态保真度越好.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_corpus import DOMAINS, load_tokens  # noqa: E402
from sunnynet.linear_baselines import LINEAR_ARCHS  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.profiling import CacheCounter, PerfTracker  # noqa: E402

BUCKETS = [(1, 16, "near"), (17, 256, "mid"), (257, 2048, "far"), (2049, 10 ** 9, "ultra")]


@torch.no_grad()
def domain_ppl(model, domain: str, seq_len: int, n_windows: int, vocab: int) -> dict:
    a = load_tokens(domain, "val")
    if a.size < seq_len + 2:
        return {"ppl": None, "tokens": 0}
    rng = np.random.default_rng(2024)
    starts = rng.integers(0, a.size - seq_len - 1, size=n_windows)
    tot, n = 0.0, 0
    model.eval()
    for s in starts:
        w = a[s:s + seq_len + 1].astype(np.int64)
        x = torch.from_numpy(w[:-1])[None, :]
        y = torch.from_numpy(w[1:])[None, :]
        logits = model(x)
        tot += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="sum"))
        n += y.numel()
    mean = tot / max(1, n)
    return {"ppl": round(math.exp(min(mean, 20)), 4), "nll": round(mean, 6), "tokens": n}


@torch.no_grad()
def long_range_recall(model, domain: str, seq_len: int, n_seqs: int, vocab: int,
                      max_distance: int = 3072, ngram: int = 1) -> dict:
    """在验证语料上统计重复上下文的 top-1 命中率, 按距离分桶.

    ngram=1: 键为单个前驱 token; ngram>1: 键为前 ngram 个 token 的片段.
    重复片段越长, 命中越说明模型真的在做长程检索而非局部平滑.
    """
    a = load_tokens(domain, "val")
    if a.size < seq_len + 2:
        return {}
    rng = np.random.default_rng(777)
    starts = rng.integers(0, a.size - seq_len - 1, size=n_seqs)
    stats = {name: [0, 0] for _, _, name in BUCKETS}
    model.eval()
    n = max(1, ngram)

    for s in starts:
        seq = a[s:s + seq_len].astype(np.int64)
        if seq.size < n + 2:
            continue
        last_seen: dict[tuple, int] = {}
        logits = model(torch.from_numpy(seq)[None, :])[0]
        for t in range(n, seq.size):
            key = tuple(int(v) for v in seq[t - n:t])
            prev = last_seen.get(key)
            if prev is not None:
                dist = t - prev
                target = int(seq[t])
                if dist <= max_distance:
                    pred = int(logits[t].argmax())
                    for lo, hi, name in BUCKETS:
                        if lo <= dist <= hi:
                            stats[name][0] += 1
                            stats[name][1] += int(pred == target)
                            break
            last_seen[key] = t

    out = {}
    for name in stats:
        hit, tot = stats[name][1], stats[name][0]
        out[name] = {"hit": hit, "total": tot,
                     "acc": round(hit / tot, 4) if tot else None}
    return out


@torch.no_grad()
def latency_profile(model, vocab: int, seq_lens=(128, 256, 512, 1024, 2048),
                    batch: int = 1, repeats: int = 3) -> list[dict]:
    """实测各序列长度的前向耗时, 用于确认线性扩展."""
    cache = CacheCounter()
    model.eval()
    rows = []
    for L in seq_lens:
        x = torch.randint(0, vocab, (batch, L))
        model(x)
        perf = PerfTracker()
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            dt = time.perf_counter() - t0
            perf.update(batch * L, dt, None, cache)
        rows.append({"seq_len": L, "ms": round(perf.ms_per_token * batch * L, 3),
                     "ms_per_token": round(perf.ms_per_token, 4),
                     "cache_hit": (round(perf.mean_cache_hit, 4)
                                   if perf.mean_cache_hit is not None else None)})
    return rows


def load_model(ckpt: str, device="cpu"):
    d = torch.load(ckpt, map_location=device, weights_only=False)
    arch = d.get("arch", "sunnynet")
    if arch == "transformer":
        from sunnynet.baseline import BaselineConfig, TransformerLM
        cfg = BaselineConfig(**d["config"])
        m = TransformerLM(cfg)
    elif arch in LINEAR_ARCHS:
        from sunnynet.linear_baselines import BLOCKS, LinearConfig, _LMBase
        cfg = LinearConfig(**d["config"])
        m = _LMBase(cfg)
        for _ in range(cfg.n_layers):
            m.blocks.append(BLOCKS[arch](cfg))
    else:
        cfg = SunnyConfig(**d["config"])
        m = SunnyNet(cfg)
    m.load_state_dict(d["model"])
    m.eval()
    return m, cfg


@torch.no_grad()
def generate(model, tk, prompt: str, max_new: int = 80, temperature: float = 0.8,
             top_k: int = 40) -> str:
    model.eval()
    ids = tk.encode(prompt)
    if not ids:
        ids = [tk.bos_id]
    s = model.init_state(1)
    x = torch.tensor([ids], dtype=torch.long)
    logits, s = model(x, return_state=True)
    out = list(ids)
    last = logits[0, -1]
    for _ in range(max_new):
        last = last / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(last, min(top_k, last.numel()))
            last = last.masked_fill(last < v[-1], float("-inf"))
        p = torch.softmax(last, dim=-1)
        nxt = int(torch.multinomial(p, 1))
        out.append(nxt)
        if nxt == tk.eos_id:
            break
        last, s = model.forward_step(torch.tensor([nxt]), s)
        last = last[0]
    return tk.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--recall-seqs", type=int, default=6)
    ap.add_argument("--ngram", type=str, default="1",
                    help="逗号分隔的 n-gram 长度, 如 1,4")
    ap.add_argument("--out", default="")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    from sunnynet.tokenizer import BPETokenizer
    tk = BPETokenizer.load(ROOT / "data" / "proc" / "tokenizer.json")
    model, cfg = load_model(args.ckpt)
    vocab = cfg.vocab_size
    if hasattr(cfg, "dim"):
        desc = f"dim={cfg.dim} layers={cfg.layers} rank={cfg.rank}"
    else:
        desc = (f"d_model={cfg.d_model} layers={cfg.n_layers} "
                f"d_ff={cfg.d_ff} heads={cfg.n_heads}")
    print(f"模型: {desc} params={model.num_params():,} "
          f"(non-embed {model.num_params(True):,})")

    res = {"ckpt": args.ckpt, "params": model.num_params(),
           "params_non_embed": model.num_params(True), "config": cfg.__dict__}

    print("\n[域困惑度]")
    res["domain_ppl"] = {}
    for d in DOMAINS:
        r = domain_ppl(model, d, args.seq_len, args.windows, vocab)
        res["domain_ppl"][d] = r
        print(f"  {d:14s} ppl={r['ppl']}  tokens={r['tokens']:,}")

    print("\n[长距离依赖: 重复上下文 top-1 命中率]")
    res["long_range"] = {}
    ngram_list = [int(x) for x in str(args.ngram).split(",") if x.strip()]
    for n in ngram_list:
        tag = f"ngram{n}"
        res["long_range"][tag] = {}
        print(f"  键长 = {n} token")
        for d in DOMAINS:
            r = long_range_recall(model, d, args.seq_len, args.recall_seqs, vocab,
                                  ngram=n)
            res["long_range"][tag][d] = r
            line = "  ".join(f"{k}={v['acc']}({v['total']})" for k, v in r.items())
            print(f"    {d:14s} {line}")

    if not args.skip_latency:
        print("\n[前向延迟 vs 序列长度]")
        rows = latency_profile(model, vocab)
        res["latency"] = rows
        for r in rows:
            print(f"  L={r['seq_len']:5d}  {r['ms']:8.2f} ms  "
                  f"{r['ms_per_token']:.4f} ms/tok  cache={r['cache_hit']}")

    if args.gen:
        print("\n[生成样例]")
        samples = {}
        for p in ["子曰：", "def quicksort(", "It was the best of times,",
                  "林黛玉道："]:
            g = generate(model, tk, p, max_new=60)
            samples[p] = g
            print(f"  >> {p}\n     {g[:180]}")
        res["samples"] = samples

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果 -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())