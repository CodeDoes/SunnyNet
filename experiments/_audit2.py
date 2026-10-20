"""应用变体后重新剖析算子构成, 定位剩余热点.

运行: python -X utf8 experiments/_audit2.py <variant> [T] [B] [threads]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402

CATS = [
    ("matmul", r"::\s*(mm|addmm|bmm|matmul|linear)\b"),
    ("layer_norm(融合)", r"layer_norm|native_layer_norm"),
    ("cumsum", r"cumsum"),
    ("exp/log", r"::\s*(exp|log)\b"),
    ("sigmoid", r"sigmoid"),
    ("var/mean/rsqrt", r"::\s*(var|mean|rsqrt)\b"),
    ("mul/add/sub/div", r"::\s*(mul|add|sub|div|neg|clamp)\b"),
    ("cat/slice/view", r"::\s*(cat|slice|narrow|select|split|view|reshape|permute|transpose|expand|as_strided|unsqueeze|squeeze)\b"),
    ("copy/fill", r"::\s*(copy_|contiguous|clone|fill_|zero_|empty|zeros|ones|detach)\b"),
    ("roll/flip", r"::\s*(roll|flip)\b"),
    ("conv", r"conv"),
    ("autograd引擎", r"engine|evaluate_function|::autograd|_local_scalar|item\b"),
    ("softmax/ce", r"softmax|cross_entropy|nll_loss"),
    ("其他", r"."),
]


def cat_of(key: str) -> str:
    for name, pat in CATS:
        if re.search(pat, key):
            return name
    return "其他"


def load_patch(variant):
    spec = importlib.util.spec_from_file_location(
        "_step_ab", str(Path(__file__).with_name("_step_ab.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VARIANTS[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("seq_len", nargs="?", type=int, default=1024)
    ap.add_argument("batch", nargs="?", type=int, default=8)
    ap.add_argument("threads", nargs="?", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    vocab, B, T = 8192, args.batch, args.seq_len
    torch.manual_seed(0)
    m = SunnyNet(SunnyConfig.small(vocab)).train()
    load_patch(args.variant)()
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        lg = m(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()

    with profile(activities=[ProfilerActivity.CPU]) as pf:
        with torch.no_grad():
            for _ in range(8):
                m(x)
    with profile(activities=[ProfilerActivity.CPU]) as ps:
        for _ in range(6):
            step()

    def agg(prof, n):
        ev = prof.key_averages()
        tot = sum(e.self_cpu_time_total for e in ev)
        cat = defaultdict(float)
        cnt = defaultdict(int)
        for e in ev:
            c = cat_of(e.key)
            cat[c] += e.self_cpu_time_total
            cnt[c] += e.count
        return {k: (v / 1000 / n, cnt[k] / n) for k, v in cat.items()}, tot / 1000 / n

    fc, ft = agg(pf, 8)
    sc, st = agg(ps, 6)
    print(f"\nvariant={args.variant}  B={B} T={T} threads={args.threads}")
    print(f"前向 {ft:.1f} ms    训练步 {st:.1f} ms")
    print(f"{'类别':18s}{'前向ms':>9s}{'cnt':>7s}{'步ms':>9s}{'cnt':>7s}{'步占比':>8s}")
    for k in sorted(set(fc) | set(sc), key=lambda k: -sc.get(k, (0, 0))[0]):
        fm, fcn = fc.get(k, (0, 0))
        sm, scn = sc.get(k, (0, 0))
        print(f"{k:18s}{fm:>9.2f}{fcn:>7.0f}{sm:>9.2f}{scn:>7.0f}{100*sm/max(st,1e-9):>7.1f}%")

    # 步 / 前向 比值
    print(f"\n反向占比 {(st-ft)/st*100:.1f}%")


if __name__ == "__main__":
    main()