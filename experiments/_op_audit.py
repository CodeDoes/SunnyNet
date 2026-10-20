"""算子级剖析: 前向与反向分别聚合, 定位冗余算子.

运行: python -X utf8 experiments/_op_audit.py [--seq-len 1024] [--batch 8] [--threads 8]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from sunnynet.baseline import solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import build_linear_baseline  # noqa: E402
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

CATS = [
    ("matmul", r"::\s*(mm|addmm|bmm|matmul|linear)\b"),
    ("cumsum", r"cumsum"),
    ("exp/log", r"::\s*(exp|log)\b"),
    ("sigmoid", r"sigmoid"),
    ("var/mean", r"::\s*(var|mean|var_mean)\b"),
    ("rsqrt/mul/add", r"::\s*(rsqrt|mul|add|sub|div|neg)\b"),
    ("cat/slice", r"::\s*(cat|slice|narrow|select|split)\b"),
    ("copy/contig", r"::\s*(copy_|contiguous|clone|to)\b"),
    ("roll/flip", r"::\s*(roll|flip)\b"),
    ("fill/zero", r"::\s*(fill_|zero_|ones|zeros|t)\b"),
    ("conv", r"::\s*(conv|convolution)\b"),
    ("relu/gelu/silu", r"::\s*(relu|gelu|silu|elu)\b"),
    ("softmax", r"softmax"),
    ("pow/sqrt", r"::\s*(pow|sqrt|abs|clamp)\b"),
    ("reshape", r"::\s*(view|reshape|permute|transpose|expand|broadcast)\b"),
    ("cumprod", r"cumprod"),
]


def cat_of(key: str) -> str:
    for name, pat in CATS:
        if re.search(pat, key):
            return name
    if "::" not in key:
        return "(python/其他)"
    return key.split("::")[1].split(".")[0][:14]


def audit(model, name, B, T, vocab, threads, with_bwd=True):
    torch.set_num_threads(threads)
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fwd():
        return model(x)

    def step():
        lg = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()

    # 前向单独
    with profile(activities=[ProfilerActivity.CPU]) as pf:
        with torch.no_grad():
            for _ in range(8):
                fwd()
    # 完整训练步
    with profile(activities=[ProfilerActivity.CPU]) as ps:
        for _ in range(6):
            step()

    def agg(prof, n):
        ev = prof.key_averages()
        tot = sum(e.self_cpu_time_total for e in ev)
        cat = defaultdict(lambda: [0.0, 0])
        for e in ev:
            c = cat_of(e.key)
            cat[c][0] += e.self_cpu_time_total
            cat[c][1] += e.count
        out = {k: (v[0] / 1000 / n, v[1] / n) for k, v in cat.items()}
        return out, tot / 1000 / n

    f_cat, f_tot = agg(pf, 8)
    s_cat, s_tot = agg(ps, 6)

    print(f"\n{'='*76}\n{name}  B={B} T={T} threads={threads}")
    print(f"前向 {f_tot:.1f} ms/次    训练步(report总) {s_tot:.1f} ms/次")
    print(f"{'类别':16s}{'前向ms':>9s}{'cnt':>7s}{'步ms':>9s}{'cnt':>7s}{'占步%':>8s}")
    keys = sorted(set(f_cat) | set(s_cat), key=lambda k: -s_cat.get(k, (0, 0))[0])
    for k in keys:
        fm, fc = f_cat.get(k, (0, 0))
        sm, sc = s_cat.get(k, (0, 0))
        print(f"{k:16s}{fm:>9.2f}{fc:>7.0f}{sm:>9.2f}{sc:>7.0f}{100*sm/max(s_tot,1e-9):>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--archs", default="sunnynet,mamba,h3,hyena,transformer,linear_attn,rwkv")
    args = ap.parse_args()

    want = args.archs.split(",")
    vocab = args.vocab

    if "sunnynet" in want:
        torch.manual_seed(0)
        m = SunnyNet(SunnyConfig.small(vocab)).train()
        audit(m, "sunnynet", args.batch, args.seq_len, vocab, args.threads)
        del m

    target = None
    if any(a != "sunnynet" for a in want):
        torch.manual_seed(0)
        ref = SunnyNet(SunnyConfig.small(vocab))
        target = ref.num_params(non_embedding=True)
        del ref

    for a in want:
        if a == "sunnynet":
            continue
        torch.manual_seed(0)
        if a == "transformer":
            from sunnynet.baseline import TransformerLM
            c, _, _ = solve_baseline_for_params(target, vocab, n_layers=6, n_heads=8, max_len=4096)
            m = TransformerLM(c).train()
        else:
            from sunnynet.linear_baselines import LINEAR_ARCHS, _ARCH_LM
            assert a in LINEAR_ARCHS
            c, _, _ = build_linear_baseline(a, target, vocab, max_len=4096, n_layers=4)
            m = _ARCH_LM[a](c).train()
        audit(m, a, args.batch, args.seq_len, vocab, args.threads)
        del m


if __name__ == "__main__":
    main()